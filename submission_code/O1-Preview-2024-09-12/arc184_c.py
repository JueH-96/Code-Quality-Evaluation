# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))

    N = len(A)
    M = 1 << 16  # Adjust M as needed to stay within time constraints

    num_trailing_zeros_parity = [0] * (2*M)
    for t in range(2*M):
        num_trailing_zeros_parity[t] = (t & -t).bit_length() % 2

    s_k = [a % (2*M) for a in A]

    f = [0] * M

    for r in range(M):
        total = 0
        for s in s_k:
            t = r + s
            if num_trailing_zeros_parity[t] % 2 == 1:
                total += 1
        f[r] = total

    max_f = max(f)
    print(max_f)


if __name__ == "__main__":
    threading.Thread(target=main).start()