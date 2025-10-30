def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # S_AB, S_AC, S_BC
    s_ab, s_ac, s_bc = data

    # older_count[X] = number of brothers older than X
    older_count = {'A': 0, 'B': 0, 'C': 0}

    # Process A vs B
    # If s_ab == '<', A is younger than B => B is older => A has one person older
    # If s_ab == '>', A is older than B => B has one person older
    if s_ab == '<':
        older_count['A'] += 1
    else:
        older_count['B'] += 1

    # Process A vs C
    if s_ac == '<':
        older_count['A'] += 1
    else:
        older_count['C'] += 1

    # Process B vs C
    if s_bc == '<':
        older_count['B'] += 1
    else:
        older_count['C'] += 1

    # The middle brother has exactly one person older than him
    for brother, cnt in older_count.items():
        if cnt == 1:
            sys.stdout.write(brother)
            return

if __name__ == "__main__":
    main()