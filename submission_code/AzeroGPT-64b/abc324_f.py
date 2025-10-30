from functools import lru_cache, partial
from heapq import heappush, heappop

class SegmentTree():
    def __init__(self, data, default=0, func=max):
        """initialize the segment tree with data"""
        self._default = default
        self._func = func
        self._len = len(data)
        self._size = _size = 1 << (self._len - 1).bit_length()

        self.data = [default] * (2 * _size)
        self.data[_size:_size + self._len] = data
        for i in reversed(range(_size)):
            self.data[i] = func(self.data[i + i], self.data[i + i + 1])

    def __delitem__(self, idx):
        self[idx] = self._default

    def __getitem__(self, idx):
        return self.data[idx + self._size]

    def __setitem__(self, idx, value):
        idx += self._size
        self.data[idx] = value
        idx >>= 1
        while idx:
            self.data[idx] = self._func(self.data[2 * idx], self.data[2 * idx + 1])
            idx >>= 1

    def __len__(self):
        return self._len

    def query(self, start, stop):
        """func of data[start, stop)"""
        start += self._size
        stop += self._size

        res_left = res_right = self._default
        while start < stop:
            if start & 1:
                res_left = self._func(res_left, self.data[start])
                start += 1
            if stop & 1:
                stop -= 1
                res_right = self._func(self.data[stop], res_right)
            start >>= 1
            stop >>= 1

        return self._func(res_left, res_right)

    def __repr__(self):
        return "SegmentTree({0})".format(self.data)

    def query_count(self, start, stop, val):
        """count of v in data[start, stop) if v <= val"""
        res = 0
        start += self._size
        stop += self._size

        while start < stop:
            if start & 1:
                res += self.data[start][val]
                start += 1
            if stop & 1:
                stop -= 1
                res += self.data[stop][val]
            start >>= 1
            stop >>= 1

        return res


N, M = map(int, input().split())

cost_list = []
edges = []
for m in range(M):
    u, v, b, c = map(int, input().split())
    edges.append((u - 1, v - 1, b, c))
    cost_list.append(c)
cost_list.sort()

reduced_cost = 0
reduced_cost_list = []
for c in cost_list:
    reduced_cost += c
    reduced_cost_list.append(reduced_cost)

data = {0: [[float('inf'), 0]]}
for idx, (u, v, b, c) in enumerate(edges):
    if not u in data: data[u] = []
    data[u].append((v, b, c, idx))

A = [0] + [-float('inf') for _ in range(N - 1)]

@lru_cache(maxsize=None)
def solve(u):
    segment = SegmentTree([-float('inf') for _ in range(M + 1)])
    best_b = 0
    best_c = 0
    
    segment[0] = (0, 0)

    for mv, mb, mc, mid in sorted(data[u]):
        inv_cost = 1 / reduced_cost_list[mid]
        dot_to_b = segment.query(0, mid + 1)
        dot_to_c = segment.query_count(0, mid + 1, mb)
        dot_val = (mb * dot_to_c + dot_to_b) * inv_cost

        best_b += mb
        best_c += mc
        segment[mid + 1] = (best_b, best_c)

        for nn in range(mv + 1, N):
            inv_cost = 1 / reduced_cost_list[nn]
            dot_to_b = segment.query(0, nn + 1)
            dot_to_c = segment.query_count(0, nn + 1, mb)
            dot_val = (mb * dot_to_c + dot_to_b) * inv_cost
            heappush(A[nn], (dot_val, (mb, nn, mc)))
            if len(A[nn]) > 10:
                heappop(A[nn])

for u in range(N):
    if u not in data:
        A[u] = []
for i, a in enumerate(A):
    for value in a:
        a[0] = value
        break
    A[i] = a

ans = -float("inf")
for v in range(N):
    while A[v]:
        mem = heappop(A[v])
        b, nn, c = mem[1]
        inv_cost = 1 / reduced_cost_list[nn]
        ans = max(ans, mem[0])
        if nn + 1 < N and nn not in data:
            heappush(A[v], (A[v + 1][0][0] + inv_cost * (A[v + 1][0][1][0] * inv_cost * (reduced_cost_list[v + 1] - reduced_cost_list[nn]) - c), (b + A[v + 1][0][1][0], v, c + A[v + 1][0][1][2])))
            heappush(A[nn + 1], (A[v + 1][0][0] + inv_cost * (A[v + 1][0][1][0] * A[v + 1][0][1][1] - A[v + 1][0][1][2]) , (b + A[v + 1][0][1][0], nn + 1, c + A[v + 1][0][1][2])))
            A[v + 1].pop(0)

print(ans)