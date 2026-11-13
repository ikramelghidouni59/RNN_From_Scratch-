import numpy as np
def rnn_cell_forward_tests(target):
  np.random.seed(1)
  xt=np.random.randn(3,10)
  a_prev=np.random.randn(5,10)
  parameters ={}
  parameters["Waa"]=np.random.randn(5,5)
  parameters["Wax"]=np.random.randn(5,3)
  parameters["Wya"]=np.random.randn(2,5)
  parameters["ba"]=np.random.randn(5,1)
  parameters["by"]=np.random.randn(2,1)
  a_next ,yt_pred ,cache =target(xt, a_prev ,parameters)
  assert a_next.shape ==(5,10),"mauvaise for a_next"
  assert yt_pred.shape ==(2,10) ,"mauvaise for yt_pred"
  assert len(cache) == 4, "le cache doit contient 4 elements "
  assert np.allclose(a_next[4],[0.59584544, 0.18141802, 0.61311866, 0.99808218, 0.85016201,0.99980978, -0.18887155, 0.99815551, 0.6531151, 0.82872037]),"value mauvaise pour a_next"
  assert np.allclose(yt_pred[1],[0.9888161, 0.01682021, 0.21140899, 0.36817467, 0.98988387,0.88945212, 0.36920224, 0.9966312, 0.9982559, 0.17746526]),"value mauvais pour yt_pred"
  print("all tests  passe")
def rnn_forward_test(target):
  np.random.seed(1)
  x=np.random.randn(3,10,4)
  a0=np.random.randn(5,10)
  parameters ={}
  parameters["Waa"]=np.random.randn(5,5)
  parameters["Wax"]=np.random.randn(5,3)
  parameters["Wya"]=np.random.randn(2,5)
  parameters["ba"]=np.random.randn(5,1)
  parameters["by"]=np.random.randn(2,1)
  a, y_pred , caches=target(x,a0 ,parameters)
  assert a.shape==(5,10,4),"mauvaise for a"
  assert y_pred.shape ==(2,10,4) ,"mauvaise for y_pred"
  assert len(caches) == 2, "le cache doit etre x "
  assert len(caches[0]) == 4, "la liste des caches doit avoir T_x elements "
  assert np.allclose(a[4][1],[-0.99999375, 0.77911235, -0.99861469, -0.99833267]), "mauvaise valeur pour a "
  assert np.allclose(y_pred[1][3],[0.79560373, 0.86224861, 0.11118257, 0.81515947]),"value mauvais pour y_pred"
  print("all tests  passe")
def lstm_cell_forward_test(target):
  np.random.seed(1)
  xt=np.random.randn(3,10)
  a_prev=np.random.randn(5,10)
  c_prev=np.random.randn(5,10)
  parameters ={}
  for g in ['f','i','o','c']:
    parameters['W'+g]=np.random.randn(5,5+3)
    parameters['b'+g]=np.random.randn(5,1)
  parameters["Wy"]=np.random.randn(2,5)
  parameters["by"]=np.random.randn(2,1)
  a_next ,c_next ,yt,cache =target(xt,a_prev,c_prev,parameters)
  assert a_next.shape ==(5,10),"mauvaise for a_next"
  assert c_next.shape ==(5,10) ,"mauvaise for c_next"
  assert yt.shape ==(2,10) ,"mauvaise for yt"
  assert len(cache) == 10, "le cache doit contient 10 elements "
  assert np.allclose(a_next[4],[-0.66408471, 0.0036921, 0.02088357, 0.22834167, -0.85575339,0.00138482, 0.76566531, 0.34631421, -0.00215674, 0.43827275]),"value mauvaise pour a_next"
  assert np.allclose(c_next[2],[0.63267805, 1.00570849, 0.35504474, 0.20690913, -1.64566718,0.11832942, 0.76449811, -0.0981561, -0.74348425, -0.26810932]),"value mauvais pour c_next"
  print("all tests  passe")
def lstm_forward_test(target):
    np.random.seed(1)
    x = np.random.randn(3,10,7)
    a0 = np.random.randn(5,10)
    parameters={}
    for g in['f','i','o','c']:
        parameters['W'+ g]=np.random.randn(5,5+3)
        parameters['b'+ g]=np.random.randn(5,1)
    parameters['Wy']=np.random.randn(2,5)
    parameters['by']=np.random.randn(2,1)

    a,y,c,caches=target(x,a0,parameters)

    assert a.shape==(5,10,7),"mauvaise for a"
    assert y.shape==(2,10,7),"mauvaise for y"
    assert c.shape==(5,10,7),"mauvaise for c"
    assert len(caches) == 2, "caches doit etre x"
    assert len(caches[0]) == 7, "nb de caches incorrect"
    assert np.isclose(a[4][3][6], 0.172117767533),"mauvaise for a"
    assert np.isclose(y[1][4][3], 0.95087346185),"mauvaise for y"
    assert np.isclose(c[1][2][1], -0.855544916718),"mauvaise for c"
    print("all tests passe")

