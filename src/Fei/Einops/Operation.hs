module Fei.Einops.Operation where

import           Control.Lens          (_1, ix, (^?!))
import           RIO
import qualified RIO.HashMap           as M

import           Fei.Einops.Expression

class (MonadIO (ExecutionMonad t), MonadThrow (ExecutionMonad t)) => TensorType t where
    type ExecutionMonad t :: * -> *

    tensorReshape :: [Int] -> t -> ExecutionMonad t t
    tensorTranpose :: [Int] -> t -> ExecutionMonad t t

rearrange :: TensorType t => t -> [Int] -> Text -> [(Text, Int)] -> ExecutionMonad t t
rearrange tensor shape expr dims =
    case parse expr of
      Left err -> throwM err
      Right e@(Expr left right) -> do
          let daxes = map (_1 %~ Axis) dims
          case solve e shape daxes of
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
