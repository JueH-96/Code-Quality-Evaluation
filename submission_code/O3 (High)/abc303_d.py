import sys

def main() -> None:
    X, Y, Z = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    INF = 10 ** 20           # large enough upper bound
    dp_off = 0               # Caps Lock light is OFF
    dp_on  = INF             # Caps Lock light is ON (impossible at start)

    for ch in S:
        new_off = INF
        new_on  = INF

        # ----- coming from “Caps OFF” -----
        if ch == 'a':
            # press only 'a'
            new_off = min(new_off, dp_off + X)
            # toggle, then press Shift+'a'
            new_on  = min(new_on,  dp_off + Z + Y)
        else:          # ch == 'A'
            # press Shift+'a'
            new_off = min(new_off, dp_off + Y)
            # toggle, then press only 'a'
            new_on  = min(new_on,  dp_off + Z + X)

        # ----- coming from “Caps ON” -----
        if ch == 'a':
            # press Shift+'a'
            new_on  = min(new_on,  dp_on + Y)
            # toggle, then press only 'a'
            new_off = min(new_off, dp_on + Z + X)
        else:          # ch == 'A'
            # press only 'a'
            new_on  = min(new_on,  dp_on + X)
            # toggle, then press Shift+'a'
            new_off = min(new_off, dp_on + Z + Y)

        dp_off, dp_on = new_off, new_on

    print(min(dp_off, dp_on))

if __name__ == "__main__":
    main()