import sys

def main():
    import sys
    input = sys.stdin.readline
    X, Y, Z = map(int, input().split())
    S = input().strip()
    INF = 10**30

    # cost_off[ch] = cost to type ch when Caps is off
    # cost_on[ch]  = cost to type ch when Caps is on
    cost_off = {'a': X, 'A': Y}
    cost_on  = {'a': Y, 'A': X}

    # dp_off = min cost so far ending with CapsLock OFF
    # dp_on  = min cost so far ending with CapsLock ON
    dp_off, dp_on = 0, INF

    for ch in S:
        new_off, new_on = INF, INF

        # Stay in OFF and type ch
        c = dp_off + cost_off[ch]
        if c < new_off:
            new_off = c
        # Toggle to ON, then type ch
        c = dp_off + Z + cost_on[ch]
        if c < new_on:
            new_on = c

        # Stay in ON and type ch
        c = dp_on + cost_on[ch]
        if c < new_on:
            new_on = c
        # Toggle to OFF, then type ch
        c = dp_on + Z + cost_off[ch]
        if c < new_off:
            new_off = c

        dp_off, dp_on = new_off, new_on

    # We don't care if caps is left on or off
    ans = dp_off if dp_off < dp_on else dp_on
    print(ans)

if __name__ == "__main__":
    main()