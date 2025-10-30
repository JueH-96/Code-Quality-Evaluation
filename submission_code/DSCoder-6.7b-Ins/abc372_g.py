import sys

def solve():
    T = int(sys.stdin.readline().strip())
    for _ in range(T):
        N = int(sys.stdin.readline().strip())
        A, B, C = [], [], []
        for _ in range(N):
            a, b, c = map(int, sys.stdin.readline().strip().split())
            A.append(a)
            B.append(b)
            C.append(c)
        A.sort()
        B.sort()
        C.sort()
        ans = 0
        for i in range(N):
            x = A[i]
            y = B[i]
            while x + y < C[i]:
                ans += 1
                x += A[i]
                y += B[i]
        print(ans)

solve()