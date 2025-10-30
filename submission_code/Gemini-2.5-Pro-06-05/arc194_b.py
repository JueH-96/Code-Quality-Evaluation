import sys

def main():
    """
    Reads input, solves the problem, and prints the output.
    """
    
    # Fast I/O
    input = sys.stdin.readline

    N = int(input())
    P = list(map(int, input().split()))

    # pos[v-1] stores the 0-indexed position of value v
    pos = [0] * N
    for i in range(N):
        pos[P[i] - 1] = i

    class FenwickTree:
        """
        A 1-indexed Fenwick Tree (or Binary Indexed Tree).
        """
        def __init__(self, size):
            self.tree = [0] * (size + 1)
            self.size = size

        def add(self, i, x):
            """Adds x to the element at index i."""
            while i <= self.size:
                self.tree[i] += x
                i += i & (-i)

        def query(self, i):
            """Queries the prefix sum up to index i (sum of [1, i])."""
            s = 0
            while i > 0:
                s += self.tree[i]
                i -= i & (-i)
            return s

    bit = FenwickTree(N)
    total_cost = 0

    # We calculate the total cost by considering elements k from N down to 1.
    # For each k, we calculate the cost to move it to its correct sorted position
    # within the subproblem of elements {1, ..., k}.
    for k in range(N, 0, -1):
        # k is the value, from N down to 1.
        
        # p is the 1-indexed original position of value k
        p = pos[k - 1] + 1

        # We use a Fenwick tree to count how many elements larger than k
        # are positioned to the left of k in the original permutation.
        # These are the elements we have already processed (from N down to k+1)
        # and whose positions have been added to the BIT.
        left_count = bit.query(p - 1)

        # c_k is the rank of k among the remaining elements {1, ..., k}.
        # This is its 1-indexed position in the conceptual array of size k.
        # It's its original position `p` minus the number of larger elements
        # that were to its left (which are now conceptually "removed").
        c_k = p - left_count
        
        # To move k from its current relative position c_k to its target
        # relative position k, it must be swapped rightwards k - c_k times.
        # The swaps occur at indices c_k, c_k+1, ..., k-1 in the conceptual
        # array of size k. The cost of these swaps sums to:
        # c_k + (c_k+1) + ... + (k-1)
        # This is calculated using the formula for the sum of an arithmetic series.
        
        sum_up_to_k_minus_1 = (k - 1) * k // 2
        sum_up_to_c_k_minus_1 = (c_k - 1) * c_k // 2
        
        cost_for_k = sum_up_to_k_minus_1 - sum_up_to_c_k_minus_1
        total_cost += cost_for_k
        
        # After processing k, add its original position to the BIT. This marks it
        # as a "large" element for subsequent (smaller) k's.
        bit.add(p, 1)

    print(total_cost)

if __name__ == "__main__":
    main()