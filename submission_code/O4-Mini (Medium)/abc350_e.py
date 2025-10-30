import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1000000)
    data = sys.stdin.read().split()
    N = int(data[0])
    A = int(data[1])
    X = float(data[2])
    Y = float(data[3])

    # We will collect all reachable states by applying floor-division by 2..6
    # Starting from N, do a DFS/BFS to discover all distinct n//b for b=2..6
    # Stop when we reach 0 (we include 0 as terminal state).
    seen = set()
    stack = [N]
    seen.add(N)
    while stack:
        n = stack.pop()
        if n == 0:
            continue
        # for each possible divisor b in 2..6 (this also covers A since A is in [2,6])
        for b in (2,3,4,5,6):
            nb = n // b
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)

    # Ensure 0 is in seen
    if 0 not in seen:
        seen.add(0)

    # Sort the states in ascending order so that when computing DP[n],
    # all DP[n//b] (b>=2) are already known (they are smaller than n).
    states = sorted(seen)

    # DP dictionary: expected cost to reduce n to 0
    DP = {}
    DP[0] = 0.0

    for n in states:
        if n == 0:
            continue
        # Option 1: deterministic division by A
        v1 = X + DP[n // A]
        # Option 2: roll die, solve the self-loop for b=1
        #    E = Y + (1/6)*( E + sum_{b=2..6} DP[n//b] )
        # => (5/6)*E = Y + (1/6)*sum_{b=2..6} DP[n//b]
        # => E = (6*Y + sum_{b=2..6} DP[n//b]) / 5
        s = 0.0
        # sum DP[n//b] for b=2..6
        # we already have DP for these since they are smaller than n
        s += DP[n // 2]
        s += DP[n // 3]
        s += DP[n // 4]
        s += DP[n // 5]
        s += DP[n // 6]
        v2 = (6.0 * Y + s) / 5.0

        DP[n] = v1 if v1 < v2 else v2

    # Print the result for the original N
    # Use high precision formatting
    ans = DP[N]
    print("{:.15f}".format(ans))


if __name__ == "__main__":
    main()