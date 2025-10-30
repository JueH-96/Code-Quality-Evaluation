def solve():
    import sys
    S_AB, S_AC, S_BC = sys.stdin.readline().split()
    
    # Initialize "scores" (the higher, the older)
    scores = {'A': 0, 'B': 0, 'C': 0}
    
    # A-B relationship
    if S_AB == '<':
        scores['B'] += 1
    else:  # S_AB == '>'
        scores['A'] += 1
    
    # A-C relationship
    if S_AC == '<':
        scores['C'] += 1
    else:  # S_AC == '>'
        scores['A'] += 1
    
    # B-C relationship
    if S_BC == '<':
        scores['C'] += 1
    else:  # S_BC == '>'
        scores['B'] += 1
    
    # Sort brothers by score ascending (lowest = youngest, highest = oldest)
    sorted_brothers = sorted(scores.items(), key=lambda x: x[1])
    
    # The second item in the sorted list is the middle brother
    print(sorted_brothers[1][0])

# Call solve() if you want to run the solution immediately
# solve()