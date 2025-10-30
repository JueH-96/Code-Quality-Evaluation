import sys
import threading

def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    # dp0 = max exp if we've defeated an even number of monsters so far
    # dp1 = max exp if we've defeated an odd number of monsters so far
    dp0 = 0
    # Use a very negative number for impossible state
    dp1 = -10**30

    for x in A:
        # If we skip this monster, parity and exp don't change:
        new_dp0 = dp0
        new_dp1 = dp1

        # If we defeat this monster:
        #  - from even count (dp0), this becomes the 1st/odd defeat: gain x, parity -> odd
        new_dp1 = max(new_dp1, dp0 + x)
        #  - from odd count (dp1), this becomes the 2nd/even defeat: gain 2*x, parity -> even
        new_dp0 = max(new_dp0, dp1 + 2*x)

        dp0, dp1 = new_dp0, new_dp1

    # Answer is the best of either parity
    ans = max(dp0, dp1)
    print(ans)

if __name__ == "__main__":
    main()