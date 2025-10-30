def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return

    s_ab, s_ac, s_bc = data[0], data[1], data[2]

    # Create a score dictionary.
    # Each brother's score will count the number of brothers they are older than.
    score = {'A': 0, 'B': 0, 'C': 0}

    # S_AB: if '<' then A is younger than B (B is older), otherwise A is older than B.
    if s_ab == '<':
        score['B'] += 1
    else:
        score['A'] += 1

    # S_AC: if '<' then A is younger than C (C is older), otherwise A is older than C.
    if s_ac == '<':
        score['C'] += 1
    else:
        score['A'] += 1

    # S_BC: if '<' then B is younger than C (C is older), otherwise B is older than C.
    if s_bc == '<':
        score['C'] += 1
    else:
        score['B'] += 1

    # In a consistent chain ordering, the oldest will have a score of 2, the youngest 0.
    # The middle brother (second oldest) must be the one with score 1.
    for brother in score:
        if score[brother] == 1:
            sys.stdout.write(brother)
            break

if __name__ == '__main__':
    main()