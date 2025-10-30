import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:2*N]))

    A.sort()
    B.sort()

    for i in range(N):
        if A[i] < B[i]:
            continue
        elif i == N-1:
            print(-1)
            return
        else:
            print(A[i])
            return

if __name__ == "__main__":
    solve()