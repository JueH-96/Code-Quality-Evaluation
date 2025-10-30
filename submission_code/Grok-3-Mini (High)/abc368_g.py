import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = [0] * (N + 1)
for i in range(1, N + 1):
    A[i] = int(data[index])
    index += 1
B = [0] * (N + 1)
for i in range(1, N + 1):
    B[i] = int(data[index])
    index += 1
Q = int(data[index])
index += 1
for _ in range(Q):
    query_type = int(data[index])
    index += 1
    if query_type == 1:
        i = int(data[index])
        x = int(data[index + 1])
        index += 2
        A[i] = x
    elif query_type == 2:
        i = int(data[index])
        x = int(data[index + 1])
        index += 2
        B[i] = x
    elif query_type == 3:
        l = int(data[index])
        r = int(data[index + 1])
        index += 2
        P = 1
        for j in range(l + 1, r + 1):
            P *= B[j]
        current_sum = 0
        current_p = P
        max_val = 0
        for k in range(l, r + 1):
            current_sum += A[k]
            value = current_sum * current_p
            if value > max_val:
                max_val = value
            if k < r:
                current_p = current_p // B[k + 1]
        print(max_val)