def min_swaps_to_make_ones_contiguous(N, S):
    # Count total number of 1s
    ones_count = S.count('1')
    if ones_count <= 1:
        return 0
    
    # Convert string to list for easier manipulation
    S = list(S)
    
    # For each possible starting position of the contiguous 1s block,
    # calculate the number of swaps needed
    min_swaps = float('inf')
    
    # Try each possible starting position
    for start in range(N - ones_count + 1):
        # Calculate swaps needed to move all 1s to positions [start, start+ones_count-1]
        swaps = 0
        # Make a copy of the string for this attempt
        current = S.copy()
        
        # Count inversions (swaps needed) to move all 1s to their target positions
        for i in range(N):
            if i >= start and i < start + ones_count:
                # This position should have 1
                if current[i] == '0':
                    # Find next 1 after this position
                    j = i + 1
                    while j < N and current[j] == '0':
                        j += 1
                    if j < N:
                        # Move this 1 to position i using adjacent swaps
                        swaps += j - i
                        # Perform the swaps
                        current[i:j+1] = ['1'] + current[i:j]
            else:
                # This position should have 0
                if current[i] == '1':
                    # Find next 0 after this position
                    j = i + 1
                    while j < N and current[j] == '1':
                        j += 1
                    if j < N:
                        # Move this 0 to position i using adjacent swaps
                        swaps += j - i
                        # Perform the swaps
                        current[i:j+1] = ['0'] + current[i:j]
        
        min_swaps = min(min_swaps, swaps)
    
    return min_swaps

# Read input
N = int(input())
S = input().strip()

# Print output
print(min_swaps_to_make_ones_contiguous(N, S))