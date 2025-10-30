def main():
    # Read inputs
    S_AB, S_AC, S_BC = input().split()
    
    # Initialize scores (how many people each brother is older than)
    scores = {'A': 0, 'B': 0, 'C': 0}
    
    # Process A-B relationship
    if S_AB == '>':  # A is older than B
        scores['A'] += 1
    else:            # B is older than A
        scores['B'] += 1
    
    # Process A-C relationship
    if S_AC == '>':  # A is older than C
        scores['A'] += 1
    else:            # C is older than A
        scores['C'] += 1
    
    # Process B-C relationship
    if S_BC == '>':  # B is older than C
        scores['B'] += 1
    else:            # C is older than B
        scores['C'] += 1
    
    # Sort brothers by their "older than" score
    # The one with the highest score is oldest, the lowest score is youngest
    # So the middle one is the second in sorted order
    sorted_brothers = sorted(scores.items(), key=lambda x: x[1])
    middle_brother = sorted_brothers[1][0]
    
    print(middle_brother)

# Do not forget to call main()
main()