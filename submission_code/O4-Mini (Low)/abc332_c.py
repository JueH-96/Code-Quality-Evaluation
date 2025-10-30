def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    S = data[2]

    max_logo_needed = 0
    i = 0
    while i < N:
        # Skip days with no plans
        if S[i] == '0':
            i += 1
            continue

        # Start of a segment without zeros
        c1 = 0  # count of '1's in this segment
        c2 = 0  # count of '2's in this segment
        while i < N and S[i] != '0':
            if S[i] == '1':
                c1 += 1
            else:  # '2'
                c2 += 1
            i += 1

        # In this segment, we can use up to M plain shirts for '1's
        used_plain = min(c1, M)
        # Remaining '1's plus all '2's must be covered by logo shirts
        logo_needed = c2 + (c1 - used_plain)
        if logo_needed > max_logo_needed:
            max_logo_needed = logo_needed

    print(max_logo_needed)


if __name__ == "__main__":
    main()