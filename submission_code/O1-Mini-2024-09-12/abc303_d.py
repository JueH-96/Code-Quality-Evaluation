# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    X, Y, Z = map(int, sys.stdin.readline().split())
    S = sys.stdin.read().strip()
    INF = 1 << 60
    dp0 = 0
    dp1 = INF
    for c in S:
        next_dp0 = INF
        next_dp1 = INF
        # From state 0
        if dp0 != INF:
            # Press 'a'
            if 'a' == c:
                if dp0 + X < next_dp0:
                    next_dp0 = dp0 + X
            # Press Shift+'a'
            if 'A' == c:
                if dp0 + Y < next_dp0:
                    next_dp0 = dp0 + Y
            # Toggle to state 1 and press 'a' or Shift+'a'
            # After toggle, state 1
            # Press 'a' in state 1: produces 'A'
            if 'A' == c:
                cost = dp0 + Z + X
                if cost < next_dp1:
                    next_dp1 = cost
            # Press Shift+'a' in state 1: produces 'a'
            if 'a' == c:
                cost = dp0 + Z + Y
                if cost < next_dp1:
                    next_dp1 = cost
        # From state 1
        if dp1 != INF:
            # Press 'a' in state 1: produces 'A'
            if 'A' == c:
                if dp1 + X < next_dp1:
                    next_dp1 = dp1 + X
            # Press Shift+'a' in state 1: produces 'a'
            if 'a' == c:
                if dp1 + Y < next_dp1:
                    next_dp1 = dp1 + Y
            # Toggle to state 0 and press 'a' or Shift+'a'
            # After toggle, state 0
            # Press 'a' in state 0: produces 'a'
            if 'a' == c:
                cost = dp1 + Z + X
                if cost < next_dp0:
                    next_dp0 = cost
            # Press Shift+'a' in state 0: produces 'A'
            if 'A' == c:
                cost = dp1 + Z + Y
                if cost < next_dp0:
                    next_dp0 = cost
        dp0, dp1 = next_dp0, next_dp1
    res = min(dp0, dp1)
    print(res)

if __name__ == "__main__":
    main()