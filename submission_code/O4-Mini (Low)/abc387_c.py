import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10000)
    L, R = map(int, sys.stdin.readline().split())

    def count_snakes_up_to(N):
        # Returns count of Snake numbers in [0, N]
        if N < 10:
            return 0
        s = list(map(int, str(N)))
        n = len(s)

        from functools import lru_cache
        @lru_cache(None)
        def dp(pos, tight, started, top_digit, length_state):
            # pos: current index in digit array
            # tight: 1 if prefix equals N's prefix, else 0
            # started: 1 if we've placed a non-zero digit already
            # top_digit: the digit of the first non-zero (the "top" digit), 0..9
            # length_state: 0 if no digits yet, 1 if exactly 1 digit placed,
            #              2 if 2 or more digits placed
            if pos == n:
                # count only if we've started and have length >= 2
                return 1 if (started and length_state == 2) else 0

            res = 0
            limit = s[pos] if tight else 9
            for d in range(limit+1):
                ntight = tight and (d == limit)
                if not started:
                    # still leading zeros
                    if d == 0:
                        # stay not started
                        res += dp(pos+1, ntight, False, 0, 0)
                    else:
                        # d becomes the top digit
                        res += dp(pos+1, ntight, True, d, 1)
                else:
                    # already have a top digit
                    # all subsequent digits must be strictly less than top_digit
                    if d >= top_digit:
                        continue
                    # update length_state
                    if length_state == 1:
                        nl = 2
                    else:
                        nl = length_state  # stays 2 if already >=2
                    res += dp(pos+1, ntight, True, top_digit, nl)
            return res

        return dp(0, True, False, 0, 0)

    ans = count_snakes_up_to(R) - count_snakes_up_to(L-1)
    print(ans)

if __name__ == "__main__":
    main()