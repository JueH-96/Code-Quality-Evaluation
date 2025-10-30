import sys

def main():
    """
    This program solves the problem by decomposing the sum and using a sweep-line algorithm with Fenwick trees.
    """
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    K = int(sys.stdin.readline())
    queries = []
    for i in range(K):
        x, y = map(int, sys.stdin.readline().split())
        queries.append((x, y, i))

    class FenwickTree:
        def __init__(self, size):
            self.tree = [0] * (size + 1)
        
        def add(self, i, delta):
            i += 1
            while i < len(self.tree):
                self.tree[i] += delta
                i += i & (-i)
        
        def query(self, i):
            if i < 0:
                return 0
            i += 1
            s = 0
            while i > 0:
                s += self.tree[i]
                i -= i & (-i)
            return s

    final_ans = [0] * K

    # The sum is |A_i - B_j|. We can split this into four parts based on comparison.
    # S = sum_{i<X,j<Y, A_i > B_j} (A_i - B_j) + sum_{i<X,j<Y, A_i <= B_j} (B_j - A_i)
    # S = (sum_{A_i>B_j} A_i) - (sum_{A_i>B_j} B_j) + (sum_{A_i<=B_j} B_j) - (sum_{A_i<=B_j} A_i)
    # Let's call them S1, S2, S3, S4. We compute each term for all queries.

    # S1 = sum_{i<X, j<Y, A_i > B_j} A_i = sum_{j<Y} sum_{i<X, A_i > B_j} A_i
    # S2 = sum_{i<X, j<Y, A_i > B_j} B_j = sum_{i<X} sum_{j<Y, B_j < A_i} B_j
    # S3 = sum_{i<X, j<Y, A_i <= B_j} B_j = sum_{i<X} sum_{j<Y, B_j >= A_i} B_j
    # S4 = sum_{i<X, j<Y, A_i <= B_j} A_i = sum_{j<Y} sum_{i<X, A_i <= B_j} A_i

    # Each term is a 2D range sum, solvable with offline sweep-line + Fenwick tree.
    # The dimensions are (index_A, index_B) with a value condition.
    # A standard way is to sweep on one index and use a BIT for the other, handling the value condition
    # by sorting events by value.

    for term_type in range(4):
        # term_type map:
        # 0: S1 (sum A_i), 1: S2 (sum B_j), 2: S3 (sum B_j), 3: S4 (sum A_i)
        
        sum_over_A = term_type in [0, 3]
        strict_ineq = term_type in [0, 1] # A > B or B < A

        # Events are points from A and B, sorted by value.
        # Queries are handled by attaching them to the sweep-line index.
        events = []
        if sum_over_A: # Sweep on B's index j
            for i in range(N):
                # A points have value A_i, index i, and weight A_i
                events.append((A[i], 0, i, A[i])) # type 0 for A point
            
            queries_by_y = [[] for _ in range(N + 1)]
            for x, y, idx in queries:
                queries_by_y[y].append((x, idx))
                
            for j in range(N):
                # B points are query points along the sweep-line
                # For strict ineq A>B, A must be strictly greater.
                # To handle ties, use a small offset or tuple sorting.
                tie_breaker = 1 if strict_ineq else -1
                events.append((B[j] + 0.5 * tie_breaker, 1, j, queries_by_y[j+1]))
        else: # Sweep on A's index i
            for j in range(N):
                events.append((B[j], 0, j, B[j])) # type 0 for B point

            queries_by_x = [[] for _ in range(N + 1)]
            for x, y, idx in queries:
                queries_by_x[x].append((y, idx))

            for i in range(N):
                tie_breaker = 1 if not strict_ineq else -1
                events.append((A[i] + 0.5 * tie_breaker, 1, i, queries_by_x[i+1]))

        events.sort()

        if sum_over_A:
            # S1 or S4
            # DS is a BIT on A's indices, storing sums of A_i
            bit = FenwickTree(N)
            for val, type, idx, payload in events:
                if type == 0: # A point
                    bit.add(idx, payload)
                else: # B point (query execution)
                    # For a fixed j, we query for various X's
                    for x_q, q_idx in payload:
                        sum_val = bit.query(x_q - 1)
                        sign = 1 if term_type == 3 else -1 # S4 vs S1
                        final_ans[q_idx] -= sign * sum_val
        else:
            # S2 or S3
            # DS are two BITs on B's indices, storing sums and counts
            bit_sum = FenwickTree(N)
            bit_count = FenwickTree(N)
            for val, type, idx, payload in events:
                if type == 0: # B point
                    bit_sum.add(idx, payload)
                    bit_count.add(idx, 1)
                else: # A point (query execution)
                    # For a fixed i, we query for various Y's
                    for y_q, q_idx in payload:
                        sum_val = bit_sum.query(y_q - 1)
                        # The total sum over values of B
                        sign = 1 if term_type == 2 else -1 # S3 vs S2
                        final_ans[q_idx] += sign * sum_val
    
    for i in range(K):
        print(final_ans[i])

if __name__ == "__main__":
    main()