# YOUR CODE HERE
def find_subsequence(A, B, start_pos, excluded_positions=None):
    """Find a subsequence of A matching B, starting from start_pos.
    If excluded_positions is provided, avoid using those positions."""
    if excluded_positions is None:
        excluded_positions = set()
    
    j = 0  # pointer for B
    positions = []
    
    for i in range(start_pos, len(A)):
        if i in excluded_positions:
            continue
        if A[i] == B[j]:
            positions.append(i)
            j += 1
            if j == len(B):
                return positions
    
    return None

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Find the first subsequence
first_subsequence = find_subsequence(A, B, 0)

if first_subsequence is None:
    print("No")
else:
    # Try to find a second subsequence
    found_second = False
    
    # Try excluding each position from the first subsequence
    for exclude_idx in range(len(first_subsequence)):
        # Create a set of positions to exclude (just one position)
        excluded = {first_subsequence[exclude_idx]}
        
        # Try to find another subsequence
        second_subsequence = find_subsequence(A, B, 0, excluded)
        
        if second_subsequence is not None:
            found_second = True
            break
    
    if found_second:
        print("Yes")
    else:
        print("No")