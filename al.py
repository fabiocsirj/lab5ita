import numpy as np

def melhor_c(A, b):
    s = len(b.shape)
    if s > 1: d = b.dot(np.linalg.inv(b.T.dot(b)))
    else:     d = b/b.T.dot(b)
    return A.T.dot(d)

def melhor_bc(A):
    b1 = A[:,0]

    for i in range(2):
        c1 = melhor_c(A, b1)

        b2 = c1
        c2 = melhor_c(A.T, b2)
        
        # print("Erro: ", np.linalg.norm(A - (c1*b1[np.newaxis].T)))

        b1 = c2
    return (b1, c1)

# A = np.random.randint(low=0, high=65, size=(254, 254))
# # b = A[:,0]
# print(A)
# print(A.shape)
# print("Norma: ", np.linalg.norm(A))

# (b, c) = melhor_bc(A)
# print("b:", b.shape)
# print("c:", c.shape)
# s = len(b.shape)
# if s > 1: x = c.dot(b.T)
# else:     x = c * b[np.newaxis].T
# e = A - x
# print("Erro:", np.linalg.norm(e))
