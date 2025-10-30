# YOUR CODE HERE
N = int(input())
S = input()

count = 0
for i in range(1, N-1):  # i from 1 to N-2 in 1-indexed terms
    # Check if seats i, i+1, i+2 match the pattern #.#
    # In 0-indexed: S[i-1], S[i], S[i+1]
    if S[i-1] == '#' and S[i] == '.' and S[i+1] == '#':
        count += 1

print(count)