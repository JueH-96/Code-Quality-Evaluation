import sys
import math

def main():
    data = sys.stdin.readline().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    # Slack for d_N: d_N <= S
    S = M - 1 - 10 * (N - 1)
    # Number of non-decreasing sequences of length N from 0..S is C(S+N, N)
    X = math.comb(S + N, N)
    out = sys.stdout
    out.write(str(X) + "
")
    # Precompute the base values for A_i = base[i] + d[i]
    base = [1 + 10 * i for i in range(N)]
    # Initialize d to all zeros
    d = [0] * N

    # Enumerate all non-decreasing sequences d of length N with values in [0..S]
    while True:
        # Print the current A sequence
        # A_i = base[i] + d[i]
        line = " ".join(str(base[i] + d[i]) for i in range(N))
        out.write(line + "
")
        # Compute next d in lex order
        for i in range(N - 1, -1, -1):
            if d[i] < S:
                v = d[i] + 1
                d[i] = v
                # Set all positions j>i to the same value to maintain non-decreasing
                for j in range(i + 1, N):
                    d[j] = v
                break
        else:
            # If no position can be increased, we're done
            break

if __name__ == "__main__":
    main()