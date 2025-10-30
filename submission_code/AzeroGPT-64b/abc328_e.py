from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import sys
from functools import cache
a = sys.stdin.buffer.read().splitlines()
N,M,K = map(int,a[0].split())
G = csr_matrix((list(map(int,a[i+1][2:].split())[-1] for i in range(M)),(list(map(int,a[i+1][2:].split())[:2]) for i in range(M))),shape=(N,N))
T = minimum_spanning_tree(G)
@cache
def f(w):
	if not w: return 0
	v = w&1
	if v: res = float('inf')
	else: res = 0
	lo,hi = T.indptr[v:v+2]
	for i in range(lo,hi):
		if not (w&T.indices[i]): continue
		nw = w^T.indices[i]
		d = T.data[i]
		if d >= K: d -= K
		res = min(res,d + f(nw))
	if res > 0: return res%K
	return res
w = (1<<N)-1
print(f(w))