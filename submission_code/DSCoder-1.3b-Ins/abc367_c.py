import sys

def solve(N, K):
    R = list(map(int, sys.stdin.readline().split()))
    R.append(K)
    R.sort()
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            if R[j] % K == 0:
                print(' '.join(map(str, R[i - 1:j])))

def main():
    N, K = map(int, sys.stdin.readline().split())
    solve(N, K)

if __name__ == "__main__":
    main()