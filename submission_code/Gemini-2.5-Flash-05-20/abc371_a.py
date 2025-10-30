# YOUR CODE HERE
def solve():
    s_ab, s_ac, s_bc = input().split()

    # Initialize scores for each brother.
    # A higher score means the brother is older than more others.
    # 0: youngest (older than none)
    # 1: middle (older than one)
    # 2: oldest (older than two)
    scores = {'A': 0, 'B': 0, 'C': 0}

    # Process S_AB relationship
    if s_ab == '<':
        # A is younger than B, so B is older than A
        scores['B'] += 1
    else: # s_ab == '>'
        # A is older than B
        scores['A'] += 1

    # Process S_AC relationship
    if s_ac == '<':
        # A is younger than C, so C is older than A
        scores['C'] += 1
    else: # s_ac == '>'
        # A is older than C
        scores['A'] += 1

    # Process S_BC relationship
    if s_bc == '<':
        # B is younger than C, so C is older than B
        scores['C'] += 1
    else: # s_bc == '>'
        # B is older than C
        scores['B'] += 1

    # Find the brother with a score of 1 (the middle brother)
    for brother, score in scores.items():
        if score == 1:
            print(brother)
            return # Found the middle brother, exit

solve()