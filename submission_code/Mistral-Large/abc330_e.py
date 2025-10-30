import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
A = list(map(int, data[2:2+N]))
queries = []
for i in range(Q):
    queries.append((int(data[2+N+2*i]), int(data[2+N+2*i+1])))

mex_set = set(A)
mex = 0
while mex in mex_set:
    mex += 1

results = []
for i, x in queries:
    i -= 1  # Convert to 0-based index
    if A[i] in mex_set:
        mex_set.remove(A[i])
    if x in mex_set:
        mex_set.remove(x)
    A[i] = x
    if x == mex:
        while mex in mex_set:
            mex += 1
    results.append(mex)

sys.stdout.write('
'.join(map(str, results)) + '
')