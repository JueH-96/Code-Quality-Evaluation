import sys

def main():
    input = sys.stdin.readline
    N, X, Y = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Find the smallest k_A such that the sum of the k_A largest A_i exceeds X
    A.sort(reverse=True)
    sa = 0
    kA = N + 1
    for i, a in enumerate(A):
        sa += a
        if sa > X:
            kA = i + 1
            break

    # Find the smallest k_B such that the sum of the k_B largest B_i exceeds Y
    B.sort(reverse=True)
    sb = 0
    kB = N + 1
    for i, b in enumerate(B):
        sb += b
        if sb > Y:
            kB = i + 1
            break

    # The answer is the minimum of these two; if neither threshold can be exceeded,
    # he eats all N dishes.
    ans = min(kA, kB)
    if ans > N:
        ans = N

    print(ans)

if __name__ == "__main__":
    main()