{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE UndecidableInstances #-}
module Fei.Einops.MXNet where

import           Control.Lens          (ix, (^?))
import           Data.Monoid           (Sum (..))
import           MXNet.Base            (DType, NDArray (..), Symbol,
                                        SymbolHandle, ndshape)
import           MXNet.Base.Tensor
import           MXNet.NN.Layer        (reshape, transpose)
import           RIO                   hiding ((^?))
import qualified RIO.HashMap           as M
import qualified RIO.List              as L (groupBy)
import qualified RIO.NonEmpty          as NE

import           Fei.Einops.Expression
import           Fei.Einops.Operation


data ReshapeOp =
      AxisKnown Int          -- a known axis
    | AxisCopy               -- copy the axis: (0,)
    | AxisSplitL Int         -- split into 2 with a known left:  (-4, n, -1)
    | AxisSplitR Int         -- split into 2 with a known right: (-4, -1, n)
    | AxisSplitM [Int]       -- split into a list of known axes: (n, m, l, ..)
    | AxisSplitU [Int] [Int] -- split into at most one unknown axis
    | AxisMerge              -- merge 2 axes: (-3,)

termsToOps :: ReshapeDirection -> HashMap Axis Int -> [Term Axis] -> Maybe [ReshapeOp]
termsToOps dir knowns terms = mapM each terms
    where
        each (Sing a) = pure $ case knowns ^? ix a of
                          Nothing -> AxisCopy
                          Just v  -> AxisKnown v
        each (Ellipse as) = let vs  = map ((knowns ^?) . ix) as
                             in case dir of
                                  ReshapeExpand -> expand vs
                                  ReshapeReduce -> reduce vs
        expand vs =
            let num = length vs
                seg = L.groupBy (\a b -> isJust a && isJust b) vs
             in case () of
                  _ | num > 2 ->
                      case seg of
                         [js] -> pure $ AxisSplitM (chopJust js)
                         [[Nothing], js] -> pure $ AxisSplitU [] (chopJust js)
                         [js, [Nothing]] -> pure $ AxisSplitU (chopJust js) []
                         [js1, [Nothing], js2] -> pure $ AxisSplitU (chopJust js1) (chopJust js2)
                         _ -> Nothing
                    | num == 2 ->
                        case vs of
                          [Nothing, Just v] -> pure $ AxisSplitR v
                          [Just v, Nothing] -> pure $ AxisSplitL v
                          [Just u, Just v]  -> pure $ AxisSplitM [u, v]
                          _                 -> Nothing
                  _ -> Nothing

        reduce vs =
            let num = length vs
                seg = L.groupBy (\a b -> isJust a && isJust b) vs
             in case () of
                  _ | num == 2 -> pure $ AxisMerge
                    | num > 2 && and (map isJust vs) -> pure $ AxisKnown $ product $ chopJust vs
                  _ -> Nothing

        chopJust js = [a | Just a <- js]

opToCode :: ReshapeOp -> [Int]
opToCode (AxisKnown v)      = [v]
opToCode AxisCopy           = [0]
opToCode (AxisSplitL v)     = [-4, v, -1]
opToCode (AxisSplitR v)     = [-4, -1, v]
opToCode (AxisSplitM vs)    = vs
opToCode (AxisSplitU us vs) = us ++ [-1] ++ vs
opToCode AxisMerge          = [-3]

numSplitU :: [ReshapeOp] -> Int
numSplitU = getSum . mconcat . map (Sum . isAxisSplitU)
    where
        isAxisSplitU (AxisSplitU _ _ ) = 1
        isAxisSplitU _                 = 0

data ReshapeSymError = CannotReshape [Term Axis] [(Axis, Int)]
    deriving Show

instance Exception ReshapeSymError

class (MonadIO (TensorMonad t), MonadThrow (TensorMonad t), PrimTensorOp t t) => MXTensor t where
    mxGetShape   :: t -> IO (Maybe [Int])
    mxReshapeSym :: ReshapeDirection -> Head Axis -> [(Axis, Int)] -> t -> TensorMonad t t

instance MXTensor SymbolHandle where
    mxGetShape _ = return Nothing
    mxReshapeSym dir (Head terms) knowns sym = do
        -- given a head, say 'c (w h)', we shall reshape a symbol of shape [c, w*h] to
        -- [c, w, h] (expand), or in the opposite direction (reduce). This is possible
        -- with special values 0, -1, -2, -3, -4 in the reshape api, see
        -- https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/symbol/symbol.html#mxnet.symbol.reshape
        --
        -- But we will restrict the forms of head.
        -- * expanding direction
        --   * multiple ellipses, each of size 2, with at least one axis known
        --   * otherwise, at most one axis unknown in all ellipses
        --
        -- * reducing direction
        --   * ellipses of size 2, with either known/unknown axes
        --   * ellipses of size > 2, with all known axes
        --
        case termsToOps dir (M.fromList knowns) terms of
          Nothing -> throwM $ CannotReshape terms knowns
          Just ops
            | numSplitU ops > 1 -> throwM $ CannotReshape terms knowns
            | otherwise -> do
                let code = concatMap opToCode ops
                reshape code sym

instance DType a => MXTensor (NDArray a) where
    mxGetShape t = ndshape t >>= return . Just . NE.toList
    mxReshapeSym = error "No need to reshape a NDArray symolically."

instance (MXTensor t, MonadIO (TensorMonad t), MonadThrow (TensorMonad t), PrimTensorOp t t) => TensorType t where
    type ExecutionMonad t = TensorMonad t
    tensorGetShape = liftIO . mxGetShape
    tensorReshapeSym = mxReshapeSym
    tensorReshape  = reshape
    tensorTranpose = flip transpose

