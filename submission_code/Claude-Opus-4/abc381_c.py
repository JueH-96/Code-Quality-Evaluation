# YOUR CODE HERE
N = int(input())
S = input()

max_length = 0

# Find all positions of '/'
for i in range(N):
    if S[i] == '/':
        # Count consecutive 1s to the left
        ones = 0
        j = i - 1
        while j >= 0 and S[j] == '1':
            ones += 1
            j -= 1
        
        # Count consecutive 2s to the right
        twos = 0
        j = i + 1
        while j < N and S[j] == '2':
            twos += 1
            j += 1
        
        # The valid 11/22 string has equal number of 1s and 2s
        # So we take the minimum of ones and twos
        valid_count = min(ones, twos)
        
        # The total length is: valid_count (1s) + 1 (/) + valid_count (2s)
        length = 2 * valid_count + 1
        
        max_length = max(max_length, length)

print(max_length)