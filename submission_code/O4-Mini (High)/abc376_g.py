import sys
import threading
import heapq

MOD = 998244353

class Item:
    __slots__ = ('v',)
    # these will be set before each test
    sumA = []
    size = []
    def __init__(self, v):
        self.v = v
    # we want the heap to pop the node with highest ratio sumA[v]/size[v]
    # so we define __lt__ inverted: self < other iff self.ratio > other.ratio
    def __lt__(self, other):
        # compare sumA[self.v] / size[self.v]  > sumA[other.v] / size[other.v]
        return Item.sumA[self.v] * Item.size[other.v] > Item.sumA[other.v] * Item.size[self.v]

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    T = int(next(it))
    out = []
    for _ in range(T):
        N = int(next(it))
        # read parents
        # p[1..N], root is 0
        children = [[] for _ in range(N+1)]
        for i in range(1, N+1):
            p = int(next(it))
            children[p].append(i)
        # read a_i
        # store in A for computing result
        A = [0] * (N+1)
        for i in range(1, N+1):
            A[i] = int(next(it))
        # build sumA and size arrays
        sumA = A[:]  # sumA[v] initialized to a[v]
        size = [0] * (N+1)
        # iterative post-order DFS to fill sumA and size
        stack = [(0, 0)]
        while stack:
            v, st = stack.pop()
            if st == 0:
                stack.append((v, 1))
                for c in children[v]:
                    stack.append((c, 0))
            else:
                sv = 1
                sa = sumA[v]
                for c in children[v]:
                    sv += size[c]
                    sa += sumA[c]
                size[v] = sv
                sumA[v] = sa
        # set class variables for comparator
        Item.sumA = sumA
        Item.size = size
        # simulate the greedy frontier search
        heap = []
        # initially the frontier is children of root (0)
        for c in children[0]:
            heapq.heappush(heap, Item(c))
        pos = [0] * (N+1)
        cnt = 0
        while heap:
            itv = heapq.heappop(heap)
            v = itv.v
            cnt += 1
            pos[v] = cnt
            # release children of v
            for c in children[v]:
                heapq.heappush(heap, Item(c))
        # compute numerator P = sum a_i * pos[i]
        P_mod = 0
        for i in range(1, N+1):
            # A[i] < MOD but pos[i] up to N
            P_mod = (P_mod + A[i] * pos[i]) % MOD
        # compute inv of total sum of A
        totalA = sumA[0]  # sumA[0] == sum of all A[i]
        invA = pow(totalA % MOD, MOD-2, MOD)
        ans = P_mod * invA % MOD
        out.append(str(ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()