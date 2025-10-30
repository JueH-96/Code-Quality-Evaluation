def minimal_operations(s, K):
    L = len(s)
    if L < K:
        return float('inf')
    pre_x = [0] * (L + 1)
    pre_dot = [0] * (L + 1)
    for i in range(L):
        pre_x[i+1] = pre_x[i] + (1 if s[i] == 'x' else 0)
        pre_dot[i+1] = pre_dot[i] + (1 if s[i] == '.' else 0)
    min_dots = float('inf')
    for j in range(L - K + 1):
        end = j + K
        x_count = pre_x[end] - pre_x[j]
        if x_count == 0:
            dot_count = pre_dot[end] - pre_dot[j]
            if dot_count < min_dots:
                min_dots = dot_count
    return min_dots

H, W, K = map(int, input().split())
S = [input().strip() for _ in range(H)]

# Generate columns
cols = []
for j in range(W):
    col = []
    for i in range(H):
        col.append(S[i][j])
    cols.append(''.join(col))

min_ops = float('inf')

# Check rows
for row in S:
    mo = minimal_operations(row, K)
    if mo < min_ops:
        min_ops = mo

# Check columns
for col in cols:
    mo = minimal_operations(col, K)
    if mo < min_ops:
        min_ops = mo

if min_ops == float('inf'):
    print(-1)
else:
    print(min_ops)