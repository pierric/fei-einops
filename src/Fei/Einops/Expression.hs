module Fei.Einops.Expression where

import           Control.Lens         (_Left, (%~))
import qualified Data.Attoparsec.Text as P
import qualified Data.Set             as Set
import           RIO                  hiding ((%~))
import qualified RIO.Text             as T

newtype Var = Var Text
    deriving (Eq, Ord, Show)
data Term = Unit | Sing Var | Ellipse [Var]
newtype Head = Head [Term]
data Expr = Expr Head Head

data Mode = ModeRearrange

data Error = DuplicatedVars Head
           | DifferentVars [Var] [Var]
           | ParsingError String

variables :: Head -> [Var]
variables (Head ts) =
    let vars Unit         = []
        vars (Sing v)     = [v]
        vars (Ellipse vs) = vs
     in concatMap vars ts

_check :: Mode -> Expr -> Either Error Expr
_check ModeRearrange expr@(Expr left right)

  | set_lv /= set_rv = Left $ DifferentVars lv rv
  | length set_lv /= Set.size set_lv = Left $ DuplicatedVars left
  | length set_rv /= Set.size set_rv = Left $ DuplicatedVars right
  | otherwise = Right expr

    where
        lv = variables left
        rv = variables right
        set_lv = Set.fromList lv
        set_rv = Set.fromList rv

parse :: Text -> Either Error Expr
parse text = P.parseOnly expr text & _Left %~ ParsingError >>= _check ModeRearrange
    where
        unit = const Unit <$> (P.string "()" <|> P.string "1")
        var  = do
            str <- liftA2 (:) P.letter (P.many' $ P.letter <|> P.digit)
            return $ Var $ T.pack str
        ellipse = do
            P.char '('
            vs <- P.sepBy1' var P.skipSpace
            P.char ')'
            return $ Ellipse vs
        head = P.sepBy1' (unit <|> (Sing <$> var) <|> ellipse) P.skipSpace
        expr = do
            h1 <- head
            P.skipSpace
            P.string "->"
            P.skipSpace
            h2 <- head
            return $ Expr (Head h1) (Head h2)
