Q = int(input())
A = []
for _ in range(Q):
    parts = input().split()
    if parts[0] == '1':
        x = int(parts[1])
        A.append(x)
    else:
        k = int(parts[1])
        idx = len(A) - k
        print(A[idx])