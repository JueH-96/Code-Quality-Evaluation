import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    Y = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]

    # Sort sweetness and saltiness in descending order
    A.sort(reverse=True)
    B.sort(reverse=True)

    # Find minimal K such that sum of top K sweetness > X
    sum_a = 0
    K_a = N + 1
    for i, a in enumerate(A):
        sum_a += a
        if sum_a > X:
            K_a = i + 1
            break

    # Find minimal K such that sum of top K saltiness > Y
    sum_b = 0
    K_b = N + 1
    for i, b in enumerate(B):
        sum_b += b
        if sum_b > Y:
            K_b = i + 1
            break

    # The earliest he can stop is the minimum of these two Ks.
    ans = min(K_a, K_b)
    # If neither exceeds by eating all, he eats all N dishes.
    if ans > N:
        ans = N

    print(ans)

if __name__ == "__main__":
    main()