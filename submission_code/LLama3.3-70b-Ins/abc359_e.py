import sys

def solve(N, H):
    A = [0] * (N + 1)
    ans = [0] * N
    step = 0
    while True:
        step += 1
        A[0] += 1
        for i in range(1, N + 1):
            if A[i - 1] > A[i] and A[i - 1] > H[i - 1]:
                A[i - 1] -= 1
                A[i] += 1
        for i in range(1, N + 1):
            if A[i] > 0 and ans[i - 1] == 0:
                ans[i - 1] = step
        if all(x > 0 for x in ans):
            break
    return ans

def main():
    input = sys.stdin.readline
    N = int(input())
    H = list(map(int, input().split()))
    ans = solve(N, H)
    print(*ans)

if __name__ == "__main__":
    main()