{-# LANGUAGE CPP #-}
module Fei.Einops(
    module Fei.Einops.Operation,
#ifdef MXNET
    module Fei.Einops.MXNet
#endif
    ) where

import           Fei.Einops.Operation
#ifdef MXNET
import           Fei.Einops.MXNet
#endif
