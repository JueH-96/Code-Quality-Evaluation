import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
Q = int(data[index + 1])
index += 2

A = list(map(int, data[index:index + N]))
index += N

mex_set = set(range(N + 1)) - set(A)
mex = min(mex_set) if mex_set else N

results = []

for _ in range(Q):
    i_k = int(data[index]) - 1
    x_k = int(data[index + 1])
    index += 2

    if A[i_k] != x_k:
        if A[i_k] == mex:
            mex_set.add(mex)
            while mex in mex_set:
                mex += 1
        elif x_k < mex:
            mex_set.add(mex)
            mex = x_k

        A[i_k] = x_k
        if x_k in mex_set:
            mex_set.remove(x_k)

    results.append(mex)

print("
".join(map(str, results)))