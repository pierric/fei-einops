{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLabels      #-}
module Fei.Einops.Operation where

import           Control.Lens          (_1, ix, (^?!))
import           GHC.OverloadedLabels  (IsLabel (..))
import           GHC.Stack
import           GHC.TypeLits          (KnownSymbol, Symbol, symbolVal)
import           RIO
import qualified RIO.HashMap           as M
import qualified RIO.Text              as T

import           Fei.Einops.Expression

data ReshapeDirection = ReshapeExpand | ReshapeReduce

data RearrangeError = RearrangeError CallStack ExprError

instance Show RearrangeError where
    show (RearrangeError cb err) = show err ++ "\n" ++ prettyCallStack cb
instance Exception RearrangeError

class (MonadIO (ExecutionMonad t), MonadThrow (ExecutionMonad t)) => TensorType t where
    type ExecutionMonad t :: * -> *

    tensorGetShape   :: t -> ExecutionMonad t (Maybe [Int])
    tensorReshape    :: [Int] -> t -> ExecutionMonad t t
    tensorReshapeSym :: ReshapeDirection -> Head Axis -> [(Axis, Int)] -> t -> ExecutionMonad t t
    tensorTranpose   :: [Int] -> t -> ExecutionMonad t t

instance KnownSymbol x => IsLabel (x :: Symbol) Axis where
    fromLabel = Axis (T.pack $ symbolVal (Proxy :: Proxy x))

infix 5 .==
(.==) :: Axis -> Int -> (Axis, Int)
(.==) = (,)

rearrange :: HasCallStack => TensorType t => t -> Text -> [(Axis, Int)] -> ExecutionMonad t t
rearrange tensor expr dims =
    case parse expr of
      Left err -> throwM $ RearrangeError callStack err
      Right e@(Expr left right) -> do
          let laxes     = axes left
              raxes     = axes right
              indices   = M.fromList $ zip laxes [0..]
              mapping   = map (\a -> indices ^?! ix a) raxes
          mshp <- tensorGetShape tensor
          case mshp of
            -- restricted mode. We can infer symbolically the relationship
            -- of axes, but it is not possible to accept the relationship by
            -- the backend. So we resort everything to the backend side.
            Nothing -> do
                tensor <- tensorReshapeSym ReshapeExpand left dims tensor
                tensor <- tensorTranpose mapping   tensor
                tensor <- tensorReshapeSym ReshapeReduce right dims tensor
                return tensor
            -- we have the shape of tensor, and we can solve the equation
            Just shape -> do
                case solve e shape dims of
                  Left err -> throwM $ RearrangeError callStack err
                  Right [h0, h1] -> do
                      let src_shape = expandHead h0
                          dst_shape = reduceHead h1
                      tensor <- tensorReshape src_shape tensor
                      tensor <- tensorTranpose mapping   tensor
                      tensor <- tensorReshape  dst_shape tensor
                      return tensor
