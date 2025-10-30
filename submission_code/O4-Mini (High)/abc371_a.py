def main():
    # Read the three comparisons
    s_ab, s_ac, s_bc = input().split()

    # older_count[X] = number of brothers X is older than
    older_count = {'A': 0, 'B': 0, 'C': 0}

    # Process A vs B
    if s_ab == '<':
        # A < B means B is older than A
        older_count['B'] += 1
    else:
        # A > B means A is older than B
        older_count['A'] += 1

    # Process A vs C
    if s_ac == '<':
        older_count['C'] += 1
    else:
        older_count['A'] += 1

    # Process B vs C
    if s_bc == '<':
        older_count['C'] += 1
    else:
        older_count['B'] += 1

    # The middle brother is the one who is older than exactly one other
    for brother in ['A', 'B', 'C']:
        if older_count[brother] == 1:
            print(brother)
            return

if __name__ == "__main__":
    main()