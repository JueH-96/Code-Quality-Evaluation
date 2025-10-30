import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    # dp0: maximum exp with an even number of defeats so far
    # dp1: maximum exp with an odd number of defeats so far
    dp0 = 0
    dp1 = -10**30  # effectively negative infinity

    for x in A:
        # skipping the monster
        ndp0 = dp0
        ndp1 = dp1

        # defeating the monster:
        # if prev count was even -> now odd, gain x points
        ndp1 = max(ndp1, dp0 + x)
        # if prev count was odd -> now even, gain 2*x points
        ndp0 = max(ndp0, dp1 + 2*x)

        dp0, dp1 = ndp0, ndp1

    # answer is best of either parity
    print(max(dp0, dp1))

if __name__ == "__main__":
    main()