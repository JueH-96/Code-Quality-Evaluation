def get_fold_pattern(pos):
    """
    Returns whether the fold at position pos is mountain (1) or valley (0)
    Uses the pattern that emerges from repeated folding
    """
    # If position is 0, it's undefined
    if pos == 0:
        return 0
    
    # Find the highest power of 2 that divides the position
    trailing_zeros = 0
    temp = pos
    while temp & 1 == 0:
        trailing_zeros += 1
        temp >>= 1
    
    # The pattern depends on whether the remaining number is congruent to 1 or 3 mod 4
    remaining = temp % 4
    
    return 1 if remaining == 1 else 0

def solve():
    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    
    # For each possible starting position i, we need to check the pattern
    # at positions i+A[k] for all k
    # We only need to check a small window of positions since the pattern repeats
    
    # The pattern repeats every 4 positions after considering trailing zeros
    # We only need to check enough positions to cover all possible combinations
    # of relative positions of A[k]
    
    max_count = 0
    # We only need to check positions up to the maximum difference between
    # any two A[k] values plus a small constant to catch all patterns
    max_diff = max(A[-1] - A[0], 16)  # 16 is enough to catch all patterns
    
    for i in range(max_diff):
        # Check if this position is valid (i + max(A) should be less than 2^100 - 1)
        if i + A[-1] >= (1 << 100) - 1:
            continue
            
        # Count mountain folds for this sequence
        count = sum(1 for k in range(N) if get_fold_pattern(i + A[k]) == 1)
        max_count = max(max_count, count)
        
        # If we've found the maximum possible count (N), we can stop
        if max_count == N:
            break
    
    print(max_count)

solve()