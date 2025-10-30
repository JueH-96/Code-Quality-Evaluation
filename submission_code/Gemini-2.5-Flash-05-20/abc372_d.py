import sys

class FenwickTree:
    def __init__(self, size):
        self.size = size
        # Fenwick tree uses 1-based indexing, so tree array size is size + 1
        self.tree = [0] * (size + 1)

    def add(self, index, value):
        """
        Adds 'value' to the element at 'index' (0-based)
        and propagates the change through the tree.
        """
        # Convert 0-based index to 1-based
        index += 1 
        while index <= self.size:
            self.tree[index] += value
            index += index & (-index)

    def query(self, index):
        """
        Returns the sum of elements from index 0 to 'index' (0-based).
        """
        # Convert 0-based index to 1-based
        index += 1
        s = 0
        while index > 0:
            s += self.tree[index]
            index -= index & (-index)
        return s

def solve():
    N = int(sys.stdin.readline())
    H = list(map(int, sys.stdin.readline().split()))

    # Step 1: Compute prev_taller[j] for all j
    # prev_taller[j] stores the index of the largest k < j such that H[k] > H[j].
    # If no such k exists, prev_taller[j] is -1.
    prev_taller = [-1] * N
    stack = [] # Stores indices k such that H[k] values are in decreasing order

    for j in range(N):
        # Pop elements from stack that are not strictly taller than H[j].
        # Since H_i != H_j for i != j, we only need to check H[stack[-1]] < H[j].
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        
        # After popping, if stack is not empty, stack.top() is the first element to the left that is taller than H[j].
        if stack:
            prev_taller[j] = stack[-1]
        
        # Push current index onto stack
        stack.append(j)

    # Step 2: Use Fenwick Tree for range updates
    # For each j, it contributes to ans[i] for i in range [max(0, prev_taller[j]), j-1].
    # This range is inclusive.
    bit = FenwickTree(N)

    for j in range(N):
        # L is the left bound for i, R is the right bound for i
        L = max(0, prev_taller[j])
        R = j - 1
        
        if L <= R: # Check if the range is valid (non-empty)
            # Increment count for i in range [L, R] by adding 1 at L and subtracting 1 at R+1
            bit.add(L, 1)
            # R+1 might be N, which is valid for Fenwick Tree of size N.
            bit.add(R + 1, -1)

    # Step 3: Query Fenwick Tree to get final counts for each i
    ans = [0] * N
    for i in range(N):
        ans[i] = bit.query(i)

    # Print results separated by spaces
    sys.stdout.write(" ".join(map(str, ans)) + "
")

# Call the solve function
if __name__ == '__main__':
    solve()