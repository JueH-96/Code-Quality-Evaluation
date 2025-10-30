def solve():
    N = int(input())
    A = []
    B = []
    C = []
    for _ in range(N):
        a, b, c = map(int, input().split())
        A.append(a)
        B.append(b)
        C.append(c)

    count = 0
    for x in range(1, 301):
        for y in range(1, 301):
            valid = True
            for i in range(N):
                if A[i] * x + B[i] * y >= C[i]:
                    valid = False
                    break
            if valid:
                count += 1
    print(count)

T = int(input())
for _ in range(T):
    solve()