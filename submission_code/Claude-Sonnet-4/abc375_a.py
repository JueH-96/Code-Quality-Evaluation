# YOUR CODE HERE
N = int(input())
S = input().strip()

count = 0
for i in range(1, N - 1):  # i from 1 to N-2 (0-indexed: 0 to N-3)
    if S[i-1] == '#' and S[i] == '.' and S[i+1] == '#':
        count += 1

print(count)