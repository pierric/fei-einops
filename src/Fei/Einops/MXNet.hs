{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE UndecidableInstances #-}
module Fei.Einops.MXNet where

import           MXNet.Base.Tensor
import           MXNet.NN.Layer       (reshape, transpose)
import           RIO

import           Fei.Einops.Operation

instance (MonadIO (TensorMonad t), MonadThrow (TensorMonad t), PrimTensorOp t t) => TensorType t where
    type ExecutionMonad t = TensorMonad t
    tensorReshape  = reshape
    tensorTranpose = flip transpose

