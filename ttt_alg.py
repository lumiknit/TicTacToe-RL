import ttt
import random

def tttAlg(game):
  # Simple strategy to win TicTacToe
  def c(i):
    # Check there are 2 same-color stones in a row
    v = max([game.board[x] for x in i])
    if v == 0: return None
    cnt = 0
    e = None
    for j in range(3):
      if v == game.board[i[j]]: cnt += 1
      else: e = i[j]
    if cnt != 2 or game.board[e] != 0: return None
    return e
  # The 1st or 2nd stone must be at the center or a corner
  if game.turns <= 1:
    if game.board[4] == 0: return 4
    else: return [0, 2, 6, 8][random.randrange(4)]
  # Otherwise, find there are 2 stones in a row
  r = c([0, 1, 2]) or c([3, 4, 5]) or c([6, 7, 8])
  r = r or c([0, 3, 6]) or c([1, 4, 7]) or c([2, 5, 8])
  r = r or c([0, 4, 8]) or c([2, 4, 6])
  if r != None: return r
  # Otherwise, just pick a random position
  for x in [4, 0, 2, 8, 6, 1, 7, 3, 5]:
    if game.board[x] == 0: return x
