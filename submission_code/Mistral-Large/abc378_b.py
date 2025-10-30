import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
index = 1
q = []
r = []
for i in range(N):
    q.append(int(data[index]))
    r.append(int(data[index + 1]))
    index += 2

Q = int(data[index])
index += 1
queries = []
for j in range(Q):
    t = int(data[index])
    d = int(data[index + 1])
    queries.append((t, d))
    index += 2

results = []
for t, d in queries:
    q_i = q[t - 1]
    r_i = r[t - 1]
    remainder = d % q_i
    if remainder < r_i:
        next_collection = d + (r_i - remainder)
    else:
        next_collection = d + (q_i - remainder + r_i)
    results.append(next_collection)

for result in results:
    print(result)