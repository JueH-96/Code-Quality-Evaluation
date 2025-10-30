# YOUR CODE HERE
import sys

def main():
    """
    This program solves the Slimes problem by iteratively finding the maximal
    absorbable range for each starting slime.

    The core idea is to find for each starting slime K, the final left and right
    boundaries (L and R) of the contiguous block of slimes it can absorb.
    A block [L, R] is absorbable by a starting slime K if for any slime at
    position i in [L, K-1], its size A[i] is smaller than the sum of slimes from
    i+1 to R. A similar condition holds for slimes to the right of K.

    These conditions can be efficiently checked and solved for using prefix sums and
    two helper arrays. Let P be the prefix sum array of A. The conditions become:
    1. For left expansion: P[i] + 2*A[i] < P[R+1] for all i in [L, K-1]
    2. For right expansion: A[j] - P[j] < -P[L] for all j in [K+1, R]

    We can find the final [L, R] for each K by starting with [K, K] and iteratively
    expanding the range based on these conditions. To make this efficient for all N
    slimes, we group slimes that have the same [L, R] at each step and update
    them together. The expansion boundaries are found using segment trees, which
    allow for logarithmic time queries. The process converges quickly to the
    final answer for all slimes.
    """
    # Set a higher recursion limit for deep segment tree traversals
    # The constraints on N can lead to deep recursion in the segment tree.
    sys.setrecursionlimit(2 * 10**6)

    # Fast I/O
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))

    # P[i] = sum(A[0]...A[i-1])
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]

    # Precompute helper arrays for the expansion conditions
    # U[i] = P[i] + 2*A[i] is used for left expansion checks.
    # W[i] = A[i] - P[i] is used for right expansion checks.
    U = [(P[i] + 2 * A[i]) for i in range(N)]
    W = [(A[i] - P[i]) for i in range(N)]

    class SegTree:
        def __init__(self, arr):
            self.n = len(arr)
            self.tree = [-(10**18)] * (4 * self.n) # Use a very small number for max
            self.arr = arr
            if self.n > 0:
                self._build(0, 0, self.n - 1)

        def _build(self, node, start, end):
            if start == end:
                self.tree[node] = self.arr[start]
            else:
                mid = (start + end) // 2
                self._build(2 * node + 1, start, mid)
                self._build(2 * node + 2, mid + 1, end)
                self.tree[node] = max(self.tree[2 * node + 1], self.tree[2 * node + 2])
        
        def _query_rightmost_ge(self, node, start, end, l, r, val):
            if r < start or l > end or self.tree[node] < val:
                return -1
            if start == end:
                return start

            mid = (start + end) // 2
            res = self._query_rightmost_ge(2 * node + 2, mid + 1, end, l, r, val)
            if res != -1:
                return res
            return self._query_rightmost_ge(2 * node + 1, start, mid, l, r, val)
        
        def find_rightmost_ge(self, l, r, val):
            if l > r: return -1
            return self._query_rightmost_ge(0, 0, self.n - 1, l, r, val)

        def _query_leftmost_ge(self, node, start, end, l, r, val):
            if r < start or l > end or self.tree[node] < val:
                return -1
            if start == end:
                return start

            mid = (start + end) // 2
            res = self._query_leftmost_ge(2 * node + 1, start, mid, l, r, val)
            if res != -1:
                return res
            return self._query_leftmost_ge(2 * node + 2, mid + 1, end, l, r, val)

        def find_leftmost_ge(self, l, r, val):
            if l > r: return -1
            return self._query_leftmost_ge(0, 0, self.n - 1, l, r, val)

    st_u = SegTree(U)
    st_w = SegTree(W)
    
    L = list(range(N))
    R = list(range(N))

    while True:
        changed = False
        groups = {}
        for k in range(N):
            bound = (L[k], R[k])
            if bound not in groups:
                groups[bound] = []
            groups[bound].append(k)

        next_L = list(L)
        next_R = list(R)

        for (l, r), indices in groups.items():
            threshold_l = P[r + 1]
            res_l = st_u.find_rightmost_ge(0, l - 1, threshold_l)
            new_l = 0 if res_l == -1 else res_l + 1
            
            threshold_r = -P[l]
            res_r = st_w.find_leftmost_ge(r + 1, N - 1, threshold_r)
            new_r = N - 1 if res_r == -1 else res_r - 1
            
            if new_l != l or new_r != r:
                changed = True
                for k in indices:
                    next_L[k] = new_l
                    next_R[k] = new_r
        
        L = next_L
        R = next_R

        if not changed:
            break

    ans = [P[R[k] + 1] - P[L[k]] for k in range(N)]
    print(*ans)

if __name__ == "__main__":
    main()