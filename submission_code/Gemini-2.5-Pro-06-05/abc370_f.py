# YOUR CODE HERE
import sys

def solve():
    """
    This function solves the Circular Cake problem by first finding the optimal
    minimum mass `x` via binary search, and then determining the number of
    uncuttable lines by analyzing all possible optimal partitions.
    """
    # Increase recursion limit for deep data structures, although the final code is iterative.
    sys.setrecursionlimit(4 * 10**5 + 5)
    
    # Read input
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    total_mass = sum(A)

    def check(m):
        """
        Checks if it's possible to partition the cake into K segments,
        each with mass at least `m`.
        """
        if m == 0: return True
        if m * K > total_mass: return False
        
        # Double the array to handle circularity
        A_double = A + A
        # Create prefix sums for efficient segment sum calculation
        P = [0] * (2 * N + 1)
        for i in range(2 * N):
            P[i+1] = P[i] + A_double[i]

        # next_start[i]: if a segment starts at i, what's the earliest start of the next segment?
        # A segment starting at i and ending at j has sum P[j+1] - P[i]
        next_start = [-1] * (2 * N + 1)
        j = 0
        for i in range(2 * N):
            while j < 2 * N and P[j+1] - P[i] < m:
                j += 1
            if j < 2 * N and P[j+1] - P[i] >= m:
                next_start[i] = j + 1
            else: # Impossible to form a valid segment from i
                next_start[i] = 2 * N + 1
        
        # Binary lifting table `jump[p][i]` stores the start of the segment after 2^p greedy jumps from i.
        max_log_k = K.bit_length()
        jump = [[2 * N + 1] * (2 * N + 2) for _ in range(max_log_k)]
        
        for i in range(2 * N + 1):
             jump[0][i] = next_start[i]

        for p in range(1, max_log_k):
            for i in range(2 * N + 2):
                jump[p][i] = jump[p-1][jump[p-1][i]]

        # For each possible starting piece `i`, find the minimum segments needed to cover N pieces.
        min_segs_needed = float('inf')
        for i in range(N):
            pos = i
            segs = 0
            for p in range(max_log_k - 1, -1, -1):
                if jump[p][pos] < i + N:
                    pos = jump[p][pos]
                    segs += 1 << p
            
            # One final jump
            pos = jump[0][pos]
            segs += 1
            
            if pos >= i + N:
                min_segs_needed = min(min_segs_needed, segs)

        return min_segs_needed <= K

    # --- Part 1: Binary search for the optimal minimum mass `x` ---
    low = 0
    high = total_mass // K + 1
    x = 0
    while low < high:
        mid = (low + high) // 2
        if mid == 0: # Base case for mass, must be positive
            low = 1
            continue
        if check(mid):
            x = mid
            low = mid + 1
        else:
            high = mid
    
    if x == 0:
        print(0, N)
        return

    # --- Part 2: Find the number of uncut lines for the optimal `x` ---
    A_double = A + A
    P = [0] * (2 * N + 1)
    for i in range(2 * N):
        P[i+1] = P[i] + A_double[i]

    next_start = [-1] * (2 * N + 1)
    j = 0
    for i in range(2 * N):
        while j < 2 * N and P[j+1] - P[i] < x:
            j += 1
        if j < 2 * N and P[j+1] - P[i] >= x:
            next_start[i] = j + 1
        else:
            next_start[i] = 2 * N + 1
    
    max_log_k = K.bit_length()
    jump = [[2 * N + 1] * (2 * N + 2) for _ in range(max_log_k)]
    for i in range(2 * N + 1):
        jump[0][i] = next_start[i]
    for p in range(1, max_log_k):
        for i in range(2 * N + 2):
            jump[p][i] = jump[p-1][jump[p-1][i]]

    def count_segments(start_node, end_node):
        if start_node >= end_node: return 0
        
        pos = start_node
        segs = 0
        for p in range(max_log_k - 1, -1, -1):
            if jump[p][pos] < end_node:
                pos = jump[p][pos]
                segs += 1 << p
        
        pos = jump[0][pos]
        segs += 1

        if pos < end_node: return float('inf')
        return segs

    # Use a difference array to mark all possible cut locations
    cuttable_diff = [0] * (2 * N + 2)
    e_ptr = 2*N
    
    # `e_max(s)` is non-increasing as `s` increases.
    # So we iterate `s` forward and move `e_ptr` backward.
    for s in range(N):
        e_min = next_start[s] - 1
        if e_min >= s + N: continue
        
        e_ptr = min(e_ptr, s + N - 1)
        while e_ptr >= e_min:
            # Check if `e_ptr` is a valid endpoint for a segment starting at `s`
            if count_segments(e_ptr + 1, s + N) <= K - 1:
                break
            e_ptr -= 1
        e_max = e_ptr
        
        if e_min <= e_max:
            cuttable_diff[e_min] += 1
            cuttable_diff[e_max + 1] -= 1
            
    cuttable = [False] * N
    current_coverage = 0
    for i in range(2 * N):
        current_coverage += cuttable_diff[i]
        if current_coverage > 0:
            cuttable[i % N] = True
            
    y = N - sum(cuttable)
    
    print(x, y)

solve()