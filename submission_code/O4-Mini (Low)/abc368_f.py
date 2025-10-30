import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    if n == 0:
        print("Bruno")
        return

    maxA = max(A)
    # Build list of proper divisors for each number up to maxA
    divisors = [[] for _ in range(maxA + 1)]
    for d in range(1, maxA // 2 + 1):
        for multiple in range(d * 2, maxA + 1, d):
            divisors[multiple].append(d)

    # Compute Grundy numbers g[x] for x=1..maxA
    g = [0] * (maxA + 1)
    # We'll use a fixed-size boolean array for mex computations
    # max number of proper divisors of any number <= 100000 is < 200
    seen = [False] * 256

    # g[1] is already 0
    for x in range(2, maxA + 1):
        # mark seen Grundy values of moves
        for d in divisors[x]:
            seen[g[d]] = True
        # find mex
        mex = 0
        while seen[mex]:
            mex += 1
        g[x] = mex
        # reset seen for next iteration
        for d in divisors[x]:
            seen[g[d]] = False

    # XOR all piles' Grundy numbers
    xor_sum = 0
    for v in A:
        xor_sum ^= g[v]

    # If XOR is non-zero, first player (Anna) wins
    print("Anna" if xor_sum != 0 else "Bruno")

if __name__ == "__main__":
    main()