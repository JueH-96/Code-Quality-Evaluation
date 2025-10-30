# YOUR CODE HERE
n = int(input())
s = input().strip()

# Count total number of 1s
ones_count = s.count('1')

# If all characters are 1s, no swaps needed
if ones_count == n:
    print(0)
else:
    min_zeros = float('inf')
    
    # Try all possible contiguous segments of length ones_count
    for i in range(n - ones_count + 1):
        # Count zeros in the segment from i to i + ones_count - 1
        zeros_in_segment = 0
        for j in range(i, i + ones_count):
            if s[j] == '0':
                zeros_in_segment += 1
        
        min_zeros = min(min_zeros, zeros_in_segment)
    
    print(min_zeros)