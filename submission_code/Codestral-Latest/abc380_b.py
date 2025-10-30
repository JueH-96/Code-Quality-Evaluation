# YOUR CODE HERE
S = input().strip()
A = []
i = 1
while i < len(S):
    count = 0
    while i < len(S) and S[i] == '-':
        count += 1
        i += 1
    A.append(count)
    i += 1  # Skip the '|'
print(' '.join(map(str, A)))