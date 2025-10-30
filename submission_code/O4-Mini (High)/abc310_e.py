def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    S = input().strip()

    # t0, t1 = number of substrings ending at the previous position
    # whose NAND‐value is 0 or 1 respectively.
    t0 = 0
    t1 = 0
    ans = 0

    for c in S:
        if c == '0':
            # Extending any substring with '0' always yields 1.
            # And the new substring (just '0') has value 0.
            new_t0 = 1
            new_t1 = t0 + t1
        else:
            # Extending substrings with value 0 by '1' → 1,
            # with value 1 by '1' → 0. And the new substr '1' → 1.
            new_t0 = t1
            new_t1 = t0 + 1

        ans += new_t1
        t0, t1 = new_t0, new_t1

    print(ans)

main()