def main():
    import sys
    input_line = sys.stdin.read().strip().split()
    if len(input_line) != 3:
        return
    s_ab, s_ac, s_bc = input_line

    # We'll use a score system.
    # For each relation, the older brother gets a point.
    scores = {'A': 0, 'B': 0, 'C': 0}

    # S_AB: if '<' then A is younger than B; so B gets a point
    if s_ab == '<':
        scores['B'] += 1
    else:  # s_ab == '>'
        scores['A'] += 1

    # S_AC: if '<' then A is younger than C; so C gets a point
    if s_ac == '<':
        scores['C'] += 1
    else:  # s_ac == '>'
        scores['A'] += 1

    # S_BC: if '<' then B is younger than C; so C gets a point
    if s_bc == '<':
        scores['C'] += 1
    else:  # s_bc == '>'
        scores['B'] += 1

    # Now, the oldest will have score 2, the middle score 1, and the youngest score 0.
    for brother, score in scores.items():
        if score == 1:
            sys.stdout.write(brother)
            break

if __name__ == '__main__':
    main()