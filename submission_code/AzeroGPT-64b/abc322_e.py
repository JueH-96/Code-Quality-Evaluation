from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import numpy as np

N, K, P = map(int, input().split())
costs = []
A_s = []
for _ in range(N):
    tmp = list(map(int, input().split()))
    costs.append(tmp[0])
    A_s.append(tmp[1:])

A_s = np.array(A_s)
b = np.full(K, P * N + 1e-10)

A_ub = -A_s
b_ub = -b
c = costs
res = linprog(c,
              A_ub=csr_matrix(A_ub),
              b_ub=b_ub,
              method='interior-point',
              bounds=(0, None),
              options={'maxiter': 10000000, 'sparse': True},
              callback=lambda x: False)

if res.success:
    ans = int(res.fun)
else:
    ans = -1

print(ans)