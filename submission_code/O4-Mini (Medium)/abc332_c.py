def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    S = data[2].strip()
    max_logo_needed = 0

    # Split the schedule by '0' days; on '0', all shirts get washed and reset.
    segments = S.split('0')
    for seg in segments:
        if not seg:
            continue
        c1 = seg.count('1')  # meal days
        c2 = seg.count('2')  # event days
        # We can cover up to M meal-days with plain shirts.
        # Any extra meal-days must use logos. Plus every event-day needs a logo.
        logos_needed = c2 + max(0, c1 - M)
        if logos_needed > max_logo_needed:
            max_logo_needed = logos_needed

    # That's the minimum number of logo T-shirts to buy.
    print(max_logo_needed)

if __name__ == "__main__":
    main()