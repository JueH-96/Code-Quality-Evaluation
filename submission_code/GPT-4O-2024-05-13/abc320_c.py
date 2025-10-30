# YOUR CODE HERE
def min_seconds_to_match(M, S1, S2, S3):
    from collections import defaultdict
    
    # Create a dictionary to store the positions of each digit in each reel
    positions = [defaultdict(list) for _ in range(3)]
    
    for i, S in enumerate([S1, S2, S3]):
        for j, char in enumerate(S):
            positions[i][char].append(j)
    
    min_time = float('inf')
    
    # Check each digit from '0' to '9'
    for digit in '0123456789':
        if digit in positions[0] and digit in positions[1] and digit in positions[2]:
            # Get the positions of the digit in each reel
            pos1 = positions[0][digit]
            pos2 = positions[1][digit]
            pos3 = positions[2][digit]
            
            # Try to find the minimum time to align all three reels
            for t1 in pos1:
                for t2 in pos2:
                    for t3 in pos3:
                        # Calculate the time needed to align all three reels
                        max_time = max(t1, t2, t3)
                        # Since the reels are cyclic, we need to account for the wrap-around
                        while max_time % M != t1 % M or max_time % M != t2 % M or max_time % M != t3 % M:
                            max_time += 1
                        min_time = min(min_time, max_time)
    
    return min_time if min_time != float('inf') else -1

# Read input
import sys
input = sys.stdin.read
data = input().split()

M = int(data[0])
S1 = data[1]
S2 = data[2]
S3 = data[3]

# Get the result
result = min_seconds_to_match(M, S1, S2, S3)

# Print the result
print(result)