def main():
    import sys
    
    S_AB, S_AC, S_BC = sys.stdin.read().strip().split()

    # Initialize scores for A, B, and C.
    scores = {'A': 0, 'B': 0, 'C': 0}

    # Update scores based on relationships.
    # A vs B
    if S_AB == '>':
        scores['A'] += 1
    else:
        scores['B'] += 1

    # A vs C
    if S_AC == '>':
        scores['A'] += 1
    else:
        scores['C'] += 1

    # B vs C
    if S_BC == '>':
        scores['B'] += 1
    else:
        scores['C'] += 1

    # Find and print the brother with score = 1 (the middle).
    for brother in ['A', 'B', 'C']:
        if scores[brother] == 1:
            print(brother)
            return

# Do not forget to call main
main()