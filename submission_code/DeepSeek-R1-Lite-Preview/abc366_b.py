N = int(input())
S = [input().strip() for _ in range(N)]
M = max(len(s) for s in S)

# Initialize T_j as lists of '*' with appropriate length
T = []

for j in range(1, M + 1):
    # Determine how many S_i have at least j characters
    num_positions = sum(1 for s in S if len(s) >= j)
    # Initialize T_j with '*' in all positions
    T_j = ['*'] * num_positions
    # Assign characters from S_i to the specified positions
    for i in range(N):
        if len(S[i]) >= j:
            pos = N - i - 1  # 0-based index for the list
            T_j[pos] = S[i][j-1]
    # Ensure that the last character is not '*'
    if T_j[-1] == '*':
        # Find the first position from the end that is not '*'
        for p in range(len(T_j)-1, -1, -1):
            if T_j[p] != '*':
                break
        else:
            # If all are '*', change the last one to 'a'
            T_j[-1] = 'a'
    T.append(''.join(T_j))

# Print the vertical strings
for t in T:
    print(t)