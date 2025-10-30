import sys

# It's a common practice in competitive programming to use fast I/O.
input = sys.stdin.readline

class FenwickTree:
    """
    A class for Fenwick Tree (Binary Indexed Tree).
    It supports point updates and prefix sum queries in O(log n) time.
    """
    def __init__(self, size):
        # The tree is 1-indexed internally for easier implementation.
        # The public interface remains 0-indexed.
        self.tree = [0] * (size + 1)
        self.size = size

    def add(self, i, x):
        """Adds x to the element at index i."""
        # Convert 0-indexed to 1-indexed
        i += 1
        while i <= self.size:
            self.tree[i] += x
            # Move to the next relevant node
            i += i & -i

    def query(self, i):
        """Queries the sum of elements from index 0 to i (inclusive)."""
        # Convert 0-indexed to 1-indexed
        i += 1
        s = 0
        while i > 0:
            s += self.tree[i]
            # Move to the parent node
            i -= i & -i
        return s

def main():
    """
    Main function to solve the problem.
    """
    try:
        N, M = map(int, input().split())
        A = list(map(int, input().split()))
    except (IOError, ValueError):
        # This problem has an empty line at the end of input for some reason.
        # This handles the case where the last read is empty.
        return

    # --- Step 1: Precompute statistics for each value in A ---
    # pos[v] will store a list of indices i where A[i] = v.
    pos = [[] for _ in range(M)]
    for i, val in enumerate(A):
        pos[val].append(i)

    # s_v[v] = count of value v in A
    # S_v[v] = sum of indices where value is v
    s_v = [0] * M
    S_v = [0] * M
    for v in range(M):
        s_v[v] = len(pos[v])
        S_v[v] = sum(pos[v])

    # --- Step 2: Calculate the initial number of inversions (for k=0) ---
    # For k=0, the sequence B is the same as A.
    # We use a Fenwick tree to count inversions in O(N log M) time.
    bit = FenwickTree(M)
    current_inversions = 0
    for i in range(N):
        val = A[i]
        # Number of elements processed so far is i.
        # Number of elements seen so far with value <= val is bit.query(val).
        # So, number of inversions with A[i] is i - bit.query(val).
        inversions_with_current_element = i - bit.query(val)
        current_inversions += inversions_with_current_element
        # Add the current element's value to the Fenwick tree.
        bit.add(val, 1)

    # --- Step 3: Use the recurrence relation to find inversions for k > 0 ---
    answers = [0] * M
    answers[0] = current_inversions

    for k in range(M - 1):
        # The value in A that results in M-1 in B(k) is v = M - 1 - k.
        # All elements A[i] = v will wrap around from M-1 to 0 in the next step.
        v = M - 1 - k
        s = s_v[v]
        S = S_v[v]
        
        # The change in the number of inversions is derived as:
        # Change = (gained inversions) - (lost inversions)
        # which simplifies to the formula below.
        change = s * (1 - N) + 2 * S
        current_inversions += change
        answers[k + 1] = current_inversions

    # --- Step 4: Print the results ---
    # Using a list and then '
'.join is faster for a large number of lines.
    output = []
    for ans in answers:
        output.append(str(ans))
    sys.stdout.write('
'.join(output) + '
')

if __name__ == "__main__":
    main()