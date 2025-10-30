import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
garbage = []
for i in range(N):
    q_i = int(data[index])
    index += 1
    r_i = int(data[index])
    index += 1
    garbage.append((q_i, r_i))

Q = int(data[index])
index += 1
queries = []
for i in range(Q):
    t_j = int(data[index])
    index += 1
    d_j = int(data[index])
    index += 1
    queries.append((t_j, d_j))

results = []
for t_j, d_j in queries:
    q, r = garbage[t_j - 1]
    if d_j % q == r:
        results.append(d_j)
    else:
        next_day = d_j + (q - d_j % q) + r
        results.append(next_day)

for result in results:
    print(result)