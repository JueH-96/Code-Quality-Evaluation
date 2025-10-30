import sys
import threading

def main():
    data = sys.stdin.readline().split()
    X = int(data[0])
    Y = int(data[1])
    Z = int(data[2])
    S = sys.stdin.readline().strip()

    # dp_off  = min time to have produced the prefix and end with CapsLock OFF
    # dp_on   = min time to have produced the prefix and end with CapsLock ON
    INF = 10**30
    dp_off = 0
    dp_on = INF

    for ch in S:
        # Before typing the next character, we may optionally toggle once
        # to get the minimal cost to be in each state.
        best_off = dp_off if dp_off < dp_on + Z else dp_on + Z
        best_on  = dp_on  if dp_on  < dp_off + Z else dp_off + Z

        # Now type ch ('a' or 'A'):
        #  - if we're OFF, plain 'a' produces 'a' at cost X, shift+'a' produces 'A' at cost Y
        #  - if we're ON, plain 'a' produces 'A' at cost X, shift+'a' produces 'a' at cost Y
        if ch == 'a':
            # to get 'a': either OFF + X, or ON + Y
            new_off = best_off + X
            new_on  = best_on  + Y
        else:  # ch == 'A'
            # to get 'A': either OFF + Y, or ON + X
            new_off = best_off + Y
            new_on  = best_on  + X

        dp_off, dp_on = new_off, new_on

    # answer is the minimum cost after producing all chars, regardless of final CapsLock state
    ans = dp_off if dp_off < dp_on else dp_on
    print(ans)

if __name__ == "__main__":
    main()