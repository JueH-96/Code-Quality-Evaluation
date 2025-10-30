# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())
    H = list(map(int, sys.stdin.readline().split()))
    queries = []
    for qid in range(Q):
        l_i, r_i = map(int, sys.stdin.readline().split())
        queries.append( (l_i, r_i, qid) )
    N = len(H)
    # Compute L_j for each building
    L = [0] * (N+1)  # 1-based indexing
    stack = []
    H = [0] + H  # 1-based indexing
    for i in range(1, N+1):
        while stack and H[stack[-1]] < H[i]:
            stack.pop()
        if stack:
            L[i] = stack[-1] + 1
        else:
            L[i] = 1
        stack.append(i)
    # Build events
    events = []
    for qid in range(len(queries)):
        l_i, r_i, idx = queries[qid]
        events.append( (-r_i, 'query', l_i, idx) )
    for j in range(1, N+1):
        events.append( (-j, 'building', L[j]) )
    events.sort()
    # Fenwick Tree
    class BIT:
        def __init__(self, n):
            self.n = n
            self.tree = [0] * (n + 2)  # 1-based indexing
        def update(self, i, delta):
            while i <= self.n:
                self.tree[i] += delta
                i += i & -i
        def query(self, i):
            res = 0
            while i > 0:
                res += self.tree[i]
                i -= i & -i
            return res
    bit = BIT(N+2)
    ans = [0] * Q
    for event in events:
        neg_r, typ = event[0], event[1]
        if typ == 'building':
            L_j = event[2]
            bit.update(L_j, 1)
        else:  # query
            l_i, idx = event[2], event[3]
            res = bit.query(l_i)
            ans[idx] = res
    for a in ans:
        print(a)
threading.Thread(target=main).start()