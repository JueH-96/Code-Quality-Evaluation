import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
B = list(map(int, data[N+1:2*N+1]))
Q = int(data[2*N+1])
queries = data[2*N+2:]

index = 0
results = []

for q in range(Q):
    query_type = int(queries[index])
    i = int(queries[index + 1])

    if query_type == 1:
        x = int(queries[index + 2])
        A[i-1] = x
        index += 3
    elif query_type == 2:
        x = int(queries[index + 2])
        B[i-1] = x
        index += 3
    elif query_type == 3:
        l = i
        r = int(queries[index + 2])
        index += 3

        max_v = 0

        for bit in range(1 << (r - l + 1)):
            v = 0
            for j in range(r - l + 1):
                if (bit >> j) & 1:
                    v += A[l - 1 + j]
                else:
                    v *= B[l - 1 + j]
            max_v = max(max_v, v)

        results.append(max_v)

sys.stdout.write("
".join(map(str, results)) + "
")