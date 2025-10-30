# YOUR CODE HERE
import sys
import threading

def main():
    import bisect
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))

    # Compute initial inversion count
    max_value = max(P)
    bit_size = max_value + 2
    bit = [0] * bit_size

    def update(i):
        while i < bit_size:
            bit[i] += 1
            i += i & -i

    def query(i):
        res = 0
        while i > 0:
            res += bit[i]
            i -= i & -i
        return res

    initial_inversions = 0
    for i in range(N):
        val = P[i]
        current_inversions = query(max_value) - query(val)
        initial_inversions += current_inversions
        update(val)

    # Build inv_counts[C]
    inv_counts = [0] * (N + 2)
    for i in range(N - 1):
        if P[i] > P[i + 1]:
            inv_counts[i + 2] += 1  # C = i+1+1

    # Build prefix_inv[C]
    prefix_inv = [0] * (N + 2)
    for C in range(2, N + 2):
        prefix_inv[C] = prefix_inv[C - 1] + inv_counts[C]

    M = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    for a in A:
        total_fixed = prefix_inv[a]
        print(initial_inversions - total_fixed)

threading.Thread(target=main).start()