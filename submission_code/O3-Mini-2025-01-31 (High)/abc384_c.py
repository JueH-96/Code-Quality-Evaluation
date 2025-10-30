def main():
    import sys

    # Read input values a, b, c, d, e for problems A, B, C, D, E respectively.
    data = sys.stdin.read().split()
    a, b, c, d, e = map(int, data)
    scores = [a, b, c, d, e]  # Mapping: index 0->A, 1->B, 2->C, 3->D, 4->E

    # Generate all 31 non-empty subsets of the 5 problems.
    # For each bitmask from 1 to 31, if the i-th bit is set, then include problem chr(ord('A') + i).
    participants = []
    for mask in range(1, 1 << 5):
        name = ""
        total = 0
        for i in range(5):
            if mask & (1 << i):
                name += chr(ord('A') + i)
                total += scores[i]
        participants.append((name, total))

    # Sort participants:
    # First, by descending score (-total)
    # In case of tie, by lexicographical order of the name (which satisfies the problem's conditions)
    participants.sort(key=lambda x: (-x[1], x[0]))

    # Print the sorted participant names.
    for p in participants:
        sys.stdout.write(p[0] + "
")

if __name__ == '__main__':
    main()