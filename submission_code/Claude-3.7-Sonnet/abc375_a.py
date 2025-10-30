# YOUR CODE HERE
N = int(input())
S = input()

count = 0
for i in range(1, N-1):
    # Check if position i and i+2 are occupied (#) and position i+1 is unoccupied (.)
    # Note: S is 0-indexed, so seat i is at S[i-1]
    if S[i-1] == '#' and S[i] == '.' and S[i+1] == '#':
        count += 1

print(count)