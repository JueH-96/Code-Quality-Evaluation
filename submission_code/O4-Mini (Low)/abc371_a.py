def main():
    import sys

    s_ab, s_ac, s_bc = sys.stdin.read().split()

    # younger[X] is the count of persons that X is younger than
    younger = {'A': 0, 'B': 0, 'C': 0}

    # Process A vs B
    if s_ab == '<':
        # A is younger than B
        younger['A'] += 1
    else:
        # B is younger than A
        younger['B'] += 1

    # Process A vs C
    if s_ac == '<':
        younger['A'] += 1
    else:
        younger['C'] += 1

    # Process B vs C
    if s_bc == '<':
        younger['B'] += 1
    else:
        younger['C'] += 1

    # The middle brother is the one who is younger than exactly one other
    for name, count in younger.items():
        if count == 1:
            print(name)
            return

if __name__ == "__main__":
    main()