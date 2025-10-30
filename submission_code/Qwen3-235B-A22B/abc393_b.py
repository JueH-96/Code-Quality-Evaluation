# Read the input string
S = input().strip()
n = len(S)
count = 0

# Iterate over all possible i and j pairs
for i in range(1, n + 1):
    for j in range(i + 1, n + 1):
        k = 2 * j - i
        if k > n:
            continue
        # Check if the characters at i, j, k are A, B, C respectively
        if S[i-1] == 'A' and S[j-1] == 'B' and S[k-1] == 'C':
            count += 1

# Output the result
print(count)