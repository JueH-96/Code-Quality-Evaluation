import sys

def main():
    N, X, Y = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Compute k_a for sweetness
    A_sorted = sorted(A, reverse=True)
    current_a = 0
    k_a = N + 1
    for i in range(N):
        current_a += A_sorted[i]
        if current_a > X:
            k_a = i + 1
            break

    # Compute k_b for saltiness
    B_sorted = sorted(B, reverse=True)
    current_b = 0
    k_b = N + 1
    for i in range(N):
        current_b += B_sorted[i]
        if current_b > Y:
            k_b = i + 1
            break

    min_k = min(k_a, k_b)
    print(min_k if min_k <= N else N)

if __name__ == "__main__":
    main()