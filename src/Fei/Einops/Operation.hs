{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLabels      #-}
module Fei.Einops.Operation where

import           Control.Lens          (_1, ix, (^?!))
import           GHC.OverloadedLabels  (IsLabel (..))
import           GHC.TypeLits          (KnownSymbol, Symbol, symbolVal)
import           RIO
import qualified RIO.HashMap           as M
import qualified RIO.Text              as T

import           Fei.Einops.Expression

class (MonadIO (ExecutionMonad t), MonadThrow (ExecutionMonad t)) => TensorType t where
    type ExecutionMonad t :: * -> *

    tensorReshape :: [Int] -> t -> ExecutionMonad t t
    tensorTranpose :: [Int] -> t -> ExecutionMonad t t

instance KnownSymbol x => IsLabel (x :: Symbol) Axis where
    fromLabel = Axis (T.pack $ symbolVal (Proxy :: Proxy x))

(.==) :: Axis -> Int -> (Axis, Int)
(.==) = (,)

rearrange :: TensorType t => t -> [Int] -> Text -> [(Axis, Int)] -> ExecutionMonad t t
rearrange tensor shape expr dims =
    case parse expr of
      Left err -> throwM err
      Right e@(Expr left right) ->
          case solve e shape dims of
            Left err -> throwM err
            Right [h0, h1] -> do
              let src_shape = expandHead h0
                  dst_shape = reduceHead h1
                  laxes     = axes left
                  raxes     = axes right
                  indices   = M.fromList $ zip laxes [0..]
                  mapping   = map (\a -> indices ^?! ix a) raxes
              tensor <- tensorReshape  src_shape tensor
              tensor <- tensorTranpose mapping   tensor
              tensor <- tensorReshape  dst_shape tensor
              return tensor
