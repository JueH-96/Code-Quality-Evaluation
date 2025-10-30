def find_middle_brother(relations):
    S_AB, S_AC, S_BC = relations
    
    # Assign scores based on age relations
    # Higher score means older brother
    score_A = 0
    score_B = 0
    score_C = 0
    
    if S_AB == "<":  # A is younger than B
        score_A -= 1
        score_B += 1
    else:  # A is older than B
        score_A += 1
        score_B -= 1
    
    if S_AC == "<":  # A is younger than C
        score_A -= 1
        score_C += 1
    else:  # A is older than C
        score_A += 1
        score_C -= 1
    
    if S_BC == "<":  # B is younger than C
        score_B -= 1
        score_C += 1
    else:  # B is older than C
        score_B += 1
        score_C -= 1
    
    # Sort brothers by score and return the middle one
    scores = [(score_A, "A"), (score_B, "B"), (score_C, "C")]
    scores.sort()
    
    return scores[1][1]  # Return the name of the middle brother

# Read input
relations = input().split()
print(find_middle_brother(relations))