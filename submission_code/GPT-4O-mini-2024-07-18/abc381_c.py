def max_11_22_length(N, S):
    max_length = 0

    for i in range(N):
        if S[i] == '/':
            # We found a '/', now we need to check for the 11/22 pattern
            left_count = 0
            right_count = 0
            
            # Count '1's to the left of '/'
            j = i - 1
            while j >= 0 and S[j] == '1':
                left_count += 1
                j -= 1
            
            # Count '2's to the right of '/'
            j = i + 1
            while j < N and S[j] == '2':
                right_count += 1
                j += 1
            
            # The total length of the valid 11/22 string centered at '/'
            total_length = 1 + left_count + right_count
            
            # Check if the total length is odd
            if total_length % 2 == 1:
                max_length = max(max_length, total_length)

    return max_length

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()
N = int(data[0])
S = data[1]

# Get the result and print it
result = max_11_22_length(N, S)
print(result)