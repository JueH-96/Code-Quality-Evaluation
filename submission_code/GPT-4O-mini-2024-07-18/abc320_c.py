def min_seconds_to_match_reels(M, S1, S2, S3):
    # Collect all possible characters that can be displayed on each reel
    possible_chars = [set(), set(), set()]
    
    for t in range(M):
        possible_chars[0].add(S1[t % M])
        possible_chars[1].add(S2[t % M])
        possible_chars[2].add(S3[t % M])
    
    # Find common characters that can be displayed on all reels
    common_chars = possible_chars[0].intersection(possible_chars[1], possible_chars[2])
    
    if not common_chars:
        return -1
    
    # Now we need to find the minimum time to display any of the common characters
    min_time = float('inf')
    
    for char in common_chars:
        # For each common character, find the minimum time to display it on all reels
        max_time = 0
        for i, S in enumerate([S1, S2, S3]):
            # Find the earliest time to display `char` on reel `i`
            earliest_time = float('inf')
            for t in range(M):
                if S[t % M] == char:
                    earliest_time = min(earliest_time, t)
            if earliest_time == float('inf'):
                max_time = float('inf')
                break
            max_time = max(max_time, earliest_time)
        
        min_time = min(min_time, max_time)
    
    return min_time if min_time != float('inf') else -1

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

M = int(data[0])
S1 = data[1]
S2 = data[2]
S3 = data[3]

# Get the result
result = min_seconds_to_match_reels(M, S1, S2, S3)

# Print the result
print(result)