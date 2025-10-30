# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    N += 0  # Ensure N is integer
    total_D = 0

    delta = [0] * (N + 2)
    for k in range(M - 1):
        a = X[k]
        b = X[k + 1]
        if a == b:
            continue
        D_k1 = (b - a + N) % N
        D_k2 = (a - b + N) % N
        D_k = min(D_k1, D_k2)
        total_D += D_k
        increase = N - 2 * D_k
        if D_k1 == D_k:
            # Short path is from a to b, moving clockwise
            start = a % N
            end = (b - 1 + N) % N
            if start <= end:
                delta[start] += increase
                delta[end + 1] -= increase
            else:
                delta[start] += increase
                delta[N] -= increase
                delta[0] += increase
                delta[end + 1] -= increase
        else:
            # Short path is from a to b, moving counter-clockwise
            start = b % N
            end = (a - 1 + N) % N
            if start <= end:
                delta[start] += increase
                delta[end + 1] -= increase
            else:
                delta[start] += increase
                delta[N] -= increase
                delta[0] += increase
                delta[end + 1] -= increase

    min_total = None
    current_delta = 0
    for e in range(N):
        current_delta += delta[e]
        total_length = total_D + current_delta
        if min_total is None or total_length < min_total:
            min_total = total_length
    print(min_total)

threading.Thread(target=main).start()