# YOUR CODE HERE
N = int(input())
S = input()

max_length = 0

# Check each '/' as a potential center
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
        
        # Calculate length of 11/22 string centered at position i
        length = 2 * min(ones, twos) + 1
        max_length = max(max_length, length)

print(max_length)