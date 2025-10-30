import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    # dp0: max XP with even number of defeated monsters so far
    # dp1: max XP with odd number of defeated monsters so far
    dp0 = 0
    dp1 = -10**30  # effectively negative infinity

    for x in A:
        # If we defeat this monster when we've defeated an odd count so far,
        # this becomes an even count, so we gain 2*x.
        # If we defeat when even so far, becomes odd, gain x.
        new_dp0 = max(dp0, dp1 + 2*x)
        new_dp1 = max(dp1, dp0 + x)
        dp0, dp1 = new_dp0, new_dp1

    # The answer is the best of having an even or odd number of defeats.
    print(max(dp0, dp1))

if __name__ == "__main__":
    main()