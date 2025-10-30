import sys
import random


def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    it = iter(data)

    N = next(it)
    Q = next(it)

    A = [next(it) for _ in range(N)]
    B = [next(it) for _ in range(N)]

    # 64-bit mask (works as modulo 2^64)
    MASK = (1 << 64) - 1

    # Fix the seed so that results are fully deterministic
    random.seed(712367)

    # Give every value (1 … N) two independent 64-bit random weights
    w1 = [0] * (N + 1)
    w2 = [0] * (N + 1)
    for v in range(1, N + 1):
        w1[v] = random.getrandbits(64)
        w2[v] = random.getrandbits(64)

    # Prefix sums of the weights for A and B (mod 2^64)
    prefA1 = [0] * (N + 1)
    prefA2 = [0] * (N + 1)
    prefB1 = [0] * (N + 1)
    prefB2 = [0] * (N + 1)

    for i in range(1, N + 1):
        a = A[i - 1]
        b = B[i - 1]

        prefA1[i] = (prefA1[i - 1] + w1[a]) & MASK
        prefA2[i] = (prefA2[i - 1] + w2[a]) & MASK
        prefB1[i] = (prefB1[i - 1] + w1[b]) & MASK
        prefB2[i] = (prefB2[i - 1] + w2[b]) & MASK

    out_lines = []
    for _ in range(Q):
        l = next(it)
        r = next(it)
        L = next(it)
        R = next(it)

        # hash of A[l..r]
        ha1 = (prefA1[r] - prefA1[l - 1]) & MASK
        ha2 = (prefA2[r] - prefA2[l - 1]) & MASK
        # hash of B[L..R]
        hb1 = (prefB1[R] - prefB1[L - 1]) & MASK
        hb2 = (prefB2[R] - prefB2[L - 1]) & MASK

        out_lines.append("Yes" if ha1 == hb1 and ha2 == hb2 else "No")

    sys.stdout.write("
".join(out_lines))


if __name__ == "__main__":
    main()