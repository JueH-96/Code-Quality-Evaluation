# YOUR CODE HERE
import sys

def solve():
    """
    Solves the problem by grouping indices by value and calculating contributions.
    The overall time complexity is O(N) and space complexity is O(N).
    """
    try:
        # Fast I/O for large inputs
        input = sys.stdin.readline
        
        N = int(input())
        A = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handles potential empty or malformed input.
        # Given the problem constraints (N>=3), this is a safeguard.
        print(0)
        return

    # Step 1: Group indices by the value in A.
    # `positions[v]` will store a list of 0-based indices where A[i] == v.
    positions = [[] for _ in range(N + 1)]
    for i, val in enumerate(A):
        positions[val].append(i)

    total_count = 0
    
    # Step 2: Iterate through each possible value (1 to N).
    for p in positions:
        m = len(p)
        
        # We need at least two occurrences of a value to form a pair (i, k)
        # such that A_i = A_k.
        if m < 2:
            continue
            
        # For a value with m occurrences at indices p_0, ..., p_{m-1}, its
        # contribution to the total count is Sum_{0<=a<b<m} (p_b - p_a - b + a).
        # This can be calculated as S1 - S2 where:
        # S1 = Sum_{0<=a<b<m} (p_b - p_a) = Sum_{k=0..m-1} (2k - m + 1) * p_k
        # S2 = Sum_{0<=a<b<m} (b - a)     = m * (m-1) * (m+1) // 6 = C(m+1, 3)

        # Calculate S1
        s1 = 0
        for k in range(m):
            s1 += (2 * k - m + 1) * p[k]
            
        # Calculate S2
        s2 = m * (m - 1) * (m + 1) // 6
        
        # Add the contribution of the current value to the total count.
        total_count += s1 - s2
        
    print(total_count)

if __name__ == "__main__":
    solve()