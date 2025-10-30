import sys

def solve():
    N = int(input().strip())
    print(N)
    for i in range(1, N+1):
        print(1, i)
    sys.stdout.flush()
    S = input().strip()
    for i in range(N):
        if S[i] == '1':
            print(i+1)
            sys.stdout.flush()
            return

solve()