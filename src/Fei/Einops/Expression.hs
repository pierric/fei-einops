{-# LANGUAGE DeriveFunctor #-}
module Fei.Einops.Expression where

import           Control.Lens         (_1, _2, _Left, ix, (%~), (^?!))
import           Control.Monad.Except (throwError)
import qualified Data.Attoparsec.Text as P
import qualified Data.Set             as Set
import qualified Math.MFSolve         as Solver
import           RIO                  hiding ((%~))
import qualified RIO.HashMap          as M
import qualified RIO.Text             as T

newtype Axis = Axis Text
    deriving (Eq, Ord, Show, Generic, Hashable)
data Term a = Sing a | Ellipse [a]
    deriving (Show, Functor)
newtype Head a = Head [Term a]
    deriving (Show, Functor)
data Expr = Expr (Head Axis) (Head Axis)
    deriving (Show)

data Mode = ModeRearrange

data Error = DuplicatedAxes (Head Axis)
           | DifferentAxes [Axis] [Axis]
           | ParsingError String
           | InvalidAxis [Axis]
           | Unsolvable (Solver.DepError Axis Float)
           | UnresolvedDim [Axis]
    deriving (Show)

instance Exception Error

isEllipse :: Term a -> Bool
isEllipse (Ellipse _) = True
isEllipse _           = False

axes :: Head a -> [a]
axes (Head ts) =
    let vars (Sing v)     = [v]
        vars (Ellipse vs) = vs
     in concatMap vars ts

_check :: Mode -> Expr -> Either Error Expr
_check ModeRearrange expr@(Expr left right)

  | set_lv /= set_rv = Left $ DifferentAxes lv rv
  | length set_lv /= Set.size set_lv = Left $ DuplicatedAxes left
  | length set_rv /= Set.size set_rv = Left $ DuplicatedAxes right
  | otherwise = Right expr

    where
        lv     = axes left
        rv     = axes right
        set_lv = Set.fromList lv
        set_rv = Set.fromList rv

parse :: Text -> Either Error Expr
parse text = P.parseOnly expr text & _Left %~ ParsingError >>= _check ModeRearrange
    where
        var  = do
            str <- liftA2 (:) P.letter (P.many' $ P.letter <|> P.digit)
            return $ Axis $ T.pack str
        ellipse = do
            P.char '('
            vs <- P.sepBy1' var P.skipSpace
            P.char ')'
            return $ Ellipse vs
        head = P.sepBy1' ((Sing <$> var) <|> ellipse) P.skipSpace
        expr = do
            h1 <- head
            P.skipSpace
            P.string "->"
            P.skipSpace
            h2 <- head
            return $ Expr (Head h1) (Head h2)

expandHead :: Head Int -> [Int]
expandHead = axes

reduceHead :: Num a => Head a -> [a]
reduceHead (Head ts) = map each ts
    where
        each (Sing a)     = a
        each (Ellipse as) = product as

makeVariables :: Expr -> HashMap Axis (Solver.Expr Axis Float)
makeVariables (Expr left right) = foldl' add M.empty all_axes
    where
        left_axes  = axes left
        right_axes = axes right
        all_axes   = left_axes ++ right_axes
        add m a = M.insertWith (flip const) a (Solver.makeVariable a) m

solve :: Expr -> [Int] -> [(Axis, Int)] -> Either Error [Head Int]
solve expr@(Expr left right) input knowns = do
    _check ModeRearrange expr
    let vars   = makeVariables expr
        wrong_kn = M.difference (M.fromList knowns) vars
    when (not $ M.null wrong_kn) $
        throwError $ InvalidAxis $ M.keys wrong_kn

    let leftEs  = (vars ^?!) . ix <$> left
        knEs   = map (_1 %~ (\k -> vars ^?! ix k)) knowns
        eq (e, i) = e Solver.=== (Solver.makeConstant $ fromIntegral i)
    solution <- _Left %~ Unsolvable $ flip Solver.execSolver Solver.noDeps $ do
        mapM_ eq $ zip (reduceHead leftEs) input
        mapM_ eq knEs
    let kwn    = M.fromList $ map (_2 %~ floor) $ Solver.knownVars solution
        leftS  = (kwn ^?!) . ix <$> left
        rightS = (kwn ^?!) . ix <$> right
        unkwn  = M.difference (() <$ vars) (() <$ kwn)
    when (not $ M.null unkwn) $
        throwError $ UnresolvedDim $ M.keys unkwn
    return [leftS, rightS]
