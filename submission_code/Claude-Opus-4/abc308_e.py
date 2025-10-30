# YOUR CODE HERE
def mex(a, b, c):
    # Find the minimum non-negative integer not in {a, b, c}
    values = {a, b, c}
    for i in range(4):  # At most we need to check 0, 1, 2, 3
        if i not in values:
            return i
    return 3  # This shouldn't happen with inputs in {0, 1, 2}

N = int(input())
A = list(map(int, input().split()))
S = input().strip()

# Precompute counts of 'E' and 'X' after each position
count_E_after = [0] * N
count_X_after = [0] * N

# Count 'X' after each position
for i in range(N-2, -1, -1):
    count_X_after[i] = count_X_after[i+1]
    if S[i+1] == 'X':
        count_X_after[i] += 1

# Count 'E' after each position
for i in range(N-2, -1, -1):
    count_E_after[i] = count_E_after[i+1]
    if S[i+1] == 'E':
        count_E_after[i] += 1

total = 0

# For each 'M' position
for i in range(N):
    if S[i] == 'M':
        # For each 'E' position after i
        for j in range(i+1, N):
            if S[j] == 'E':
                # For each 'X' position after j
                for k in range(j+1, N):
                    if S[k] == 'X':
                        total += mex(A[i], A[j], A[k])

print(total)