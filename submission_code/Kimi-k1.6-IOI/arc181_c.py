n = int(input())
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

# Create rank arrays for P and Q
rank_P = [0] * (n + 1)  # rank_P[x] gives the 1-based index of x in P
for idx in range(n):
    x = P[idx]
    rank_P[x] = idx + 1

rank_Q = [0] * (n + 1)  # rank_Q[y] gives the 1-based index of y in Q
for idx in range(n):
    y = Q[idx]
    rank_Q[y] = idx + 1

# Generate the grid
for i in range(1, n + 1):
    row = []
    for j in range(1, n + 1):
        r = rank_P[i]
        c = rank_Q[j]
        if r + c > n:
            row.append('1')
        else:
            row.append('0')
    print(''.join(row))