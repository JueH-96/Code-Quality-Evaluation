def can_make_equal_sums(N, A, B):
    # Count number of -1s in each sequence
    a_minus_ones = A.count(-1)
    b_minus_ones = B.count(-1)
    
    # Get non-negative values
    a_fixed = [x for x in A if x != -1]
    b_fixed = [x for x in B if x != -1]
    
    # If all pairs are -1, we can make them equal
    if a_minus_ones == N and b_minus_ones == N:
        return True
        
    # If all elements are fixed (no -1s), check if we can rearrange A to make sums equal
    if a_minus_ones == 0 and b_minus_ones == 0:
        # Sort A's elements to try all possible arrangements
        a_fixed.sort()
        # Try all permutations implicitly by checking if any sum exists
        sums = set()
        for i in range(N):
            sums.add(A[i] + B[i])
        return len(sums) == 1
    
    # Get min and max possible sums from fixed elements
    min_sum = float('inf')
    max_sum = float('-inf')
    
    # For positions where neither is -1
    for i in range(N):
        if A[i] != -1 and B[i] != -1:
            curr_sum = A[i] + B[i]
            min_sum = min(min_sum, curr_sum)
            max_sum = max(max_sum, curr_sum)
    
    # If we found fixed sums and they're different, it's impossible
    if min_sum != float('inf') and min_sum != max_sum:
        return False
    
    # If we have no fixed sums, we can make all sums equal
    if min_sum == float('inf'):
        return True
    
    target_sum = min_sum  # All pairs must sum to this
    
    # Check if we can achieve target_sum for each position
    for i in range(N):
        if A[i] != -1 and B[i] != -1:
            if A[i] + B[i] != target_sum:
                return False
        elif A[i] == -1 and B[i] != -1:
            if B[i] > target_sum:  # Can't make negative A[i]
                return False
        elif A[i] != -1 and B[i] == -1:
            if A[i] > target_sum:  # Can't make negative B[i]
                return False
    
    return True

# Read input
N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Print result
print("Yes" if can_make_equal_sums(N, A, B) else "No")