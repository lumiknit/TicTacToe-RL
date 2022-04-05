import signal
import sys

# Import numpy
import numpy as np

# Import torch
import torch
import torch.nn as nn
import torch.optim as optim

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# Import tictactoe
import ttt
import ttt_alg

# CONST
MODEL_PATH = "./TTT_MODEL"

# -- TTT Game Helpers

def gameToInput(game):
  # Change a game into np array, and add some noise
  n = game.toNumpy(game.turn)
  r = np.random.rand(3, 3, 3) / 10.0
  return torch.tensor((n + r), dtype=torch.double) \
      .reshape(3 * 3 * 3).to(device)

def pickRandomMove(game):
  # Pick random empty cell
  t = None
  v = np.random.rand()
  idx = np.random.randint(9)
  for off in range(9):
    rdx = (idx + off) % 9
    if game.board[rdx] == 0:
      t = rdx
      g = game.clone()
      g.place(rdx)
      if g.checkWin() == None: break
  return t

def cutFoul(game, qvs):
  # Fill -100 into cells which is not empty
  q = qvs.clone().detach()
  for i in range(9):
    if game.board[i] != 0:
      q[i] = -100.0
  return q

# -- DQN

model = nn.Sequential(
    nn.Linear(3 * 3 * 3, 243).double(),
    nn.ReLU(),
    nn.Linear(243, 36).double(),
    nn.ReLU(),
    nn.Linear(36, 36).double(),
    nn.ReLU(),
    nn.Linear(36, 3 * 3).double(),
).to(device)

lossFn = nn.MSELoss()

gamma = 0.95

def calcQ(game, idx):
  # Calculate reward obtained when a stone is put on idx cell
  if not game.place(idx): # Foul: Q = -1
    return -1.0, game.turn
  else:
    w = game.checkWin()
    if w == None: # Not finished: Q = - max_a Q(s', a)
      with torch.no_grad():
        qvs2 = cutFoul(game, model(gameToInput(game)))
      return (0 - gamma * torch.max(qvs2).item()), w
    elif w == 0: # Draw: Q = 0
      return 0.0, w
    elif w == 3 - game.turn: # Win: Q = 1
      return 1.0, w
    else: # Lose: Q = -1
      return -1.0, w

# Learning loop condition
learning_running = False

def learningSignalHandler(sig, frame):
  # When SIGINT is raised, stop learning
  global learning_running
  print("Learning will be stopped!")
  signal.signal(signal.SIGINT, signal.SIG_DFL)
  learning_running = False

def learn():
  # Deep learning on Q-NN

  global learning_running

  learning_rate = 0.02

  # I don't know why Adam optimizer does not work well
  # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
  # Use SGD+Momentum optimizer
  optimizer = optim.SGD(
      model.parameters(),
      lr=learning_rate,
      momentum=0.9,
      weight_decay=1e-5)

  # Exploration parameter
  epsilon = 0.25

  # # of epochs
  n_epoch = 100000
  # Epoch when epsilon start to decrease
  dec_ep_epoch = 4000
  # Epsilon decreasement factor
  ep_dim = 0.99999
  # Lower-bound of epsilon
  ep_lb = 0.05
  # Interval of printing learning state
  print_interval = 200

  # Loss statistics
  n_loss = 0
  acc_loss = 0

  # Set signal handler to allow to stop leraning
  signal.signal(signal.SIGINT, learningSignalHandler)
  print("Press ^C to stop learning")

  # Start running
  learning_running = True
  for epoch in range(n_epoch):
    if not learning_running: break
    # Make a new game board
    game = ttt.Game()
    while True:
      # Calculate good answer by my strategy
      alg_idx = ttt_alg.tttAlg(game)
      # Calculate Q-val
      qvs = model(gameToInput(game))
      # Make an expected Q vector
      Y = qvs.clone().detach()
      # Explore all cells
      for idx in range(9):
        # Put a stone if we can
        if game.board[idx] != 0: continue
        g = game.clone()
        # State transition & calculate reward
        nq, w = calcQ(g, idx)
        # Put a reward into an expected Q vector
        Y[idx] = nq
      # Back propagation
      optimizer.zero_grad()
      loss = lossFn(qvs, Y)
      loss.backward()
      optimizer.step()
      # Update loss
      acc_loss += loss.item()
      n_loss += 1
      # Put a stone
      # If rand() < epsilon, go to random
      # if rand() < 2 epsilon, use my strategy
      # otherwise, use Q-NN
      idx = None
      with torch.no_grad():
        qvs = cutFoul(game, model(gameToInput(game)))
      rnd = np.random.rand()
      if rnd < epsilon:
        idx = pickRandomMove(game)
      elif rnd < 2 * epsilon:
        idx = ttt_alg.tttAlg(game)
      else:
        idx = torch.argmax(qvs).item()
      # Put a stone and check the game is done
      nq, w = calcQ(game, idx)
      if w != None:
        break
    # Print loss statistics
    if epoch % print_interval == print_interval - 1:
      print("{:6d}: e={:.4f}; L = {:8.4f}" \
          .format(epoch + 1, epsilon, acc_loss / n_loss))
      acc_loss = 0
      n_loss = 0
    # Decrease epsilon
    if epoch > dec_ep_epoch:
      epsilon *= ep_dim
      if epsilon < ep_lb: epsilon = ep_lb

# -- Play game

def modelAction(game):
  # Play TTT with Q-NN
  global model
  with torch.no_grad():
    qv = cutFoul(game, model(gameToInput(game)))
  print(qv)
  return torch.argmax(qv).item()

def randAction(game):
  # Play TTT randomly
  idx = np.random.randint(9)
  for off in range(9):
    rdx = (idx + off) % 9
    if game.board[rdx] == 0: return rdx
  return idx

def runGame(a, b):
  g = ttt.Game()
  # Choose who goes first
  sw = False
  if np.random.rand() < 0.5:
    sw = False
    g.play(a, b)
  else:
    sw = True
    g.play(b, a)
  # Check winner
  w = g.checkWin()
  if w != None and w >= 1 and sw: w = 3 - w
  return w

if __name__ == "__main__":
  msg = "(q)uit/(s)ave/(l)oad/(L)earn" + \
      "/vs (r)andom/vs (a)i/vs (A)lgo/(R)andom test" + \
      "/(p)lay"
  while True:
    print('==========================================')
    x = input(msg + "> ")
    if x == 'q': break
    elif x == 's': # Save model
      torch.save(model, MODEL_PATH)
    elif x == 'l': # Load model
      model = torch.load(MODEL_PATH)
      model.eval()
    elif x == 'L': # Learn
      learn()
    elif x == 'r': # model vs random
      runGame(modelAction, randAction)
    elif x == 'a': # model vs model
      runGame(modelAction, modelAction)
    elif x == 'A': # model vs my strategy
      runGame(modelAction, ttt_alg.tttAlg)
    elif x == 'R': # Calculate winning rate of model vs random
      count = 0
      win = 0
      for i in range(100):
        count += 1
        res = runGame(modelAction, randAction)
        if res == 1: win += 1
        elif res == 0: win += 0.5
      print("Model win rate {:.4f} ({} / {})".format(
        win / count, win, count))
    elif x == 'P': # model vs human
      runGame(modelAction, None)

