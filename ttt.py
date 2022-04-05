import numpy as np

class Game:
  def __init__(self):
    self.board = [0] * 9
    self.turn = 1
    self.turns = 0
    self.foul = False

  def clone(self):
    g = Game()
    for i in range(9):
      g.board[i] = self.board[i]
    g.turn = self.turn
    g.turns = self.turns
    return g

  def rotate(self):
    # Clone and rotate clockwise
    g = Game()
    for r in range(3):
      for c in range(3):
        g.board[3 * r + c] = self.board[3 * (2 - c) + r]
    g.turn = self.turn
    g.turns = self.turns
    return g

  def place(self, idx):
    # Place stone at board[idx]
    # Iff the player cannot place a stone at idx, return False
    if self.board[idx] != 0:
      self.foul = True
      return False
    self.board[idx] = self.turn
    self.turn = 3 - self.turn
    self.turns += 1
    return True

  def check(self, i0, i1, i2):
    # Check whether 3 stones of a same color are placed in i0, i1, i2
    v = self.board[i0]
    if 0 != v and v == self.board[i1] and v == self.board[i2]:
      return v
    else: return None

  def checkWin(self):
    # If some player 1 or 2 win, return 1 or 2
    # If they draw, return 0
    # Otherwise return None
    if self.foul:
      return 3 - self.turn
    r = None
    r = r or self.check(0, 1, 2) or self.check(3, 4, 5) or self.check(6, 7, 8)
    r = r or self.check(0, 3, 6) or self.check(1, 4, 7) or self.check(2, 5, 8)
    r = r or self.check(0, 4, 8) or self.check(2, 4, 6)
    if self.turns >= 9: r = r or 0
    return r
  
  def toNumpy(self, player):
    # Create 3 * 3 * 3 np array
    a = []
    for i in range(3):
      p = i
      if player == 2: p = (3 - p) % 3
      b = []
      for r in range(3):
        col = []
        for c in range(3):
          col.append(1. if self.board[r * 3 + c] == p else 0.)
        b.append(col)
      a.append(b)
    return np.array(a)

  def toS(self):
    ch_arr = ['.', 'O', 'X']
    s = "---\nTurn: {} ({})\n".format(self.turn, ch_arr[self.turn])
    s += " | 0 1 2\n-+------"
    for r in range(3):
      s += "\n{}| ".format(r * 3)
      for c in range(3):
        s += "{} ".format(ch_arr[self.board[r * 3 + c]])
    return s

  def play(self, f1, f2):
    # Play game using two input functions
    if f1 == None: f1 = userInput
    if f2 == None: f2 = userInput
    tfn = [None, f1, f2]
    while self.turns < 9: # While empty cell exists
      print(self.toS())
      # Take an index of cell to put a stone
      x = tfn[self.turn](self)
      print("{}P INPUT = {}".format(self.turn, x))
      # Check foul move.
      if type(x) != int or x < 0 or x >= 9 or self.board[x] != 0:
        print(self.toS())
        print("Player {} put a stone on a non-empty cell!, Player {} win!" \
            .format(self.turn, 3 - self.turn))
        return
      # Place a stone
      self.place(x)
      # Check game is done
      w = self.checkWin()
      if w == 0: # Draw
        print(self.toS())
        print("Draw")
        return
      elif w != None: # Player `w` win
        print(self.toS())
        print("Player {} win!".format(w))
        return
    print(self.toS())
    print("Draw")

def userInput(game):
  i = None
  while i == None:
    try:
      # Read a index of cell (0~8)
      x = input("Idx(0~8) > ")
      i = int(x)
    except:
      print("Wrong Input!")
  return i

if __name__ == "__main__":
  # Play game with user inputs
  game = Game()
  game.play(None, None)
