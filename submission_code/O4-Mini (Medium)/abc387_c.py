import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10000)
    from functools import lru_cache

    data = sys.stdin.read().strip().split()
    L = int(data[0])
    R = int(data[1])

    def count_snake(n):
        # count snake numbers in [0..n]
        s = list(map(int, str(n)))
        length = len(s)

        @lru_cache(None)
        def dp(pos, tight, started, second, top, mmax):
            # pos: current digit index
            # tight: is prefix equal to n so far
            # started: have we seen first non-zero digit (top digit)
            # second: have we placed at least one digit after top
            # top: value of the top digit (first non-zero)
            # mmax: maximum of all digits after top so far
            if pos == length:
                # accept if we have at least two digits and top > every other
                if started and second and top > mmax:
                    return 1
                else:
                    return 0
            res = 0
            up = s[pos] if tight else 9
            for d in range(up + 1):
                ntight = tight and (d == up)
                nstarted = started or (d != 0)
                if not nstarted:
                    # still leading zeros
                    res += dp(pos+1, ntight, False, False, 0, 0)
                else:
                    if not started and d != 0:
                        # this digit is the top digit
                        ntop = d
                        nsecond = False
                        nmmax = 0
                    else:
                        # this digit is after top
                        ntop = top
                        # once started was True, we are in the "other digits"
                        nsecond = second or started
                        # update max of other digits
                        nmmax = mmax if started else 0
                        # note: if started==False, that means this is the first non-zero, but we've
                        # handled that in the previous branch. So here started==True.
                        nmmax = max(nmmax, d)
                    res += dp(pos+1, ntight, True, nsecond, ntop, nmmax)
            return res

        return dp(0, True, False, False, 0, 0)

    ans = count_snake(R)
    if L > 0:
        ans -= count_snake(L-1)
    print(ans)

if __name__ == "__main__":
    main()