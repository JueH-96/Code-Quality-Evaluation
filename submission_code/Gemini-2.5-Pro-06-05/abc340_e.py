import sys

# It's good practice to use fast I/O in competitive programming
input = sys.stdin.readline

class BIT:
    """
    Fenwick Tree (Binary Indexed Tree) implementation.
    This implementation uses 0-based indexing for its public methods,
    while the internal tree structure is 1-based.
    """
    def __init__(self, n):
        self.size = n
        self.tree = [0] * (n + 1)

    def add(self, i, x):
        """Adds x to the element at 0-indexed i."""
        i += 1
        while i <= self.size:
            self.tree[i] += x
            i += i & -i

    def query(self, i):
        """Queries the prefix sum up to 0-indexed i."""
        i += 1
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        line = input()
        if not line: return
        N, M = map(int, line.split())
        A = list(map(int, input().split()))
        B = list(map(int, input().split()))
    except (IOError, ValueError):
        return

    # total_global_add tracks balls added to all boxes from full cycles.
    total_global_add = 0
    # emptied_balls tracks the total count of balls removed from each box.
    emptied_balls = [0] * N
    
    # The BIT manages additions from the 'remainder' distributions.
    # It needs to handle updates at index N, so its size is N+1.
    bit = BIT(N + 1)

    for b in B:
        # Calculate current number of balls K in box b.
        # This is the initial count + global additions + specific range additions - balls taken out.
        delta_b = bit.query(b)
        K = A[b] + total_global_add + delta_b - emptied_balls[b]
        
        if K <= 0:
            continue
            
        # Record the removal of K balls from box b.
        emptied_balls[b] += K

        # Distribute the K balls.
        full_cycles = K // N
        remainder = K % N
        
        # Add balls from full cycles to all boxes via the global counter.
        total_global_add += full_cycles
        
        if remainder > 0:
            # Distribute the remaining balls to boxes [(b+1)%N, ..., (b+remainder)%N].
            # This is a range update.
            l = (b + 1) % N
            r = (b + remainder) % N
            
            if l <= r:
                # Standard range update: add 1 to [l, r].
                # Implemented by adding 1 at index l and -1 at index r+1.
                bit.add(l, 1)
                bit.add(r + 1, -1)
            else:
                # Wrap-around range update: add 1 to [l, N-1] and [0, r].
                # This can be implemented as two range updates:
                # 1. Add 1 to [0, r] -> bit.add(0, 1), bit.add(r+1, -1)
                # 2. Add 1 to [l, N-1] -> bit.add(l, 1) (implicit end at N-1)
                bit.add(0, 1)
                bit.add(r + 1, -1)
                bit.add(l, 1)

    # Calculate final ball counts for each box.
    final_result = [0] * N
    for j in range(N):
        delta_j = bit.query(j)
        final_count = A[j] + total_global_add + delta_j - emptied_balls[j]
        final_result[j] = final_count

    print(*final_result)

if __name__ == "__main__":
    solve()