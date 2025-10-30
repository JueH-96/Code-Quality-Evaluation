# YOUR CODE HERE
S = input().strip()
n = len(S)
count = 0

# For each position j (1-indexed)
for j in range(1, n + 1):
    if S[j-1] == 'B':
        # For each possible interval d
        for d in range(1, min(j, n - j + 1)):
            i = j - d
            k = j + d
            
            # Check if i and k are valid positions
            if 1 <= i <= n and 1 <= k <= n:
                # Check if the characters match
                if S[i-1] == 'A' and S[k-1] == 'C':
                    count += 1

print(count)