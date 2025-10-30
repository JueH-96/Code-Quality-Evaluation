import sys
input = sys.stdin.readline

# Get input
s = list(input().strip())

# Preprocess: Get distance from key A to all other keys assuming s is sorted.
max_dist_to_key = sorted(range(26))
min_dist_to_key = max_dist_to_key[::-1]

# Calculate total distance
total_distance = 0
for i, k in enumerate(s):
    index_in_sorted = sorted(s).index(k)
    min_dist = min_dist_to_key[index_in_sorted]
    max_dist = max_dist_to_key[index_in_sorted]
    
    # Determine the optimal distance based on the current position
    if i == 0:
        total_distance += min_dist
    else:
        
        prev_index_in_sorted = sorted(s).index(s[i-1])
        if prev_index_in_sorted < index_in_sorted:
            # Moving to the right
            if (2 * i - 1) < (max_dist - index_in_sorted):
                total_distance += min_dist
            else:
                total_distance += max_dist
        else:
            # Moving to the left
            if (2 * (25 - i) - 1) < (index_in_sorted - min_dist):
                total_distance += min_dist
            else:
                total_distance += max_dist

# Print the answer
print(total_distance)