# YOUR CODE HERE
import sys
import heapq

# Set a higher recursion limit for safety, though the solution is iterative.
# sys.setrecursionlimit(4 * 10**5 + 5)

MOD = 998244353

class Node:
    """A custom class for priority queue items to handle fractional priorities and tie-breaking."""
    def __init__(self, u, sum_a, size):
        self.u = u
        self.sum_a = sum_a
        self.size = size
    
    def __lt__(self, other):
        """
        Comparison for the min-heap. We want to pop the node with the highest priority,
        so the highest priority node should be considered the "smallest".
        Priority is defined by the ratio sum_a/size (descending).
        The tie-breaker is the node index `u` (ascending).
        `a < b` is true if `a` has higher priority than `b`.
        """
        # Compare ratios using cross-multiplication to avoid floating-point arithmetic.
        val = self.sum_a * other.size - other.sum_a * self.size
        if val != 0:
            return val > 0
        return self.u < other.u

def solve():
    """Solves a single test case."""
    try:
        N_str = sys.stdin.readline()
        if not N_str: return
        N = int(N_str)
        
        if N == 0:
            # Constraints say N>=1, but as a safeguard.
            if len(sys.stdin.readline().split()) > 0: # p
                sys.stdin.readline() # a
            sys.stdout.write('0
')
            return

        p = list(map(int, sys.stdin.readline().split()))
        a = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        return

    a_full = [0] + a
    
    children = [[] for _ in range(N + 1)]
    for i in range(N):
        children[p[i]].append(i + 1)

    size = [0] * (N + 1)
    sum_a_sub = [0] * (N + 1)

    # A post-order traversal can be done iteratively from N down to 0
    # because the parent's index is always smaller than its child's index.
    for i in range(N, -1, -1):
        s = 1
        sa = a_full[i]
        for c in children[i]:
            s += size[c]
            sa += sum_a_sub[c]
        size[i] = s
        sum_a_sub[i] = sa

    total_a = sum_a_sub[0]

    # The simulation using a priority queue
    pq = []
    for c in children[0]:
        heapq.heappush(pq, Node(c, sum_a_sub[c], size[c]))
    
    total_wct = 0
    time = 0
    while pq:
        time += 1
        node = heapq.heappop(pq)
        u = node.u
        
        total_wct = (total_wct + a_full[u] * time)
        if total_wct >= MOD:
            total_wct %= MOD
        
        for c in children[u]:
            heapq.heappush(pq, Node(c, sum_a_sub[c], size[c]))

    inv_total_a = pow(total_a, MOD - 2, MOD)
    expected_value = (total_wct * inv_total_a) % MOD
    sys.stdout.write(str(expected_value) + '
')

def main():
    """Main function to handle multiple test cases."""
    try:
        T_str = sys.stdin.readline()
        if not T_str: return
        T = int(T_str)
        for _ in range(T):
            solve()
    except (IOError, ValueError):
        return

if __name__ == "__main__":
    main()