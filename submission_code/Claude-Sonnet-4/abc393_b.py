# YOUR CODE HERE
S = input().strip()
n = len(S)
count = 0

# Convert to 0-indexed for easier array access
for i in range(n):
    if S[i] == 'A':
        for j in range(i + 1, n):
            if S[j] == 'B':
                # Calculate k based on equal interval condition
                # In 1-indexed: j - i = k - j, so k = 2j - i
                # In 0-indexed: (j+1) - (i+1) = (k+1) - (j+1), so k = 2j - i
                k = 2 * j - i
                if k < n and k > j and S[k] == 'C':
                    count += 1

print(count)