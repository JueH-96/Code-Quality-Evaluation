def main():
    # Read input scores for problems A, B, C, D, E
    # The input is a single line with 5 space-separated integers.
    scores_input = list(map(int, input().split()))
    
    # problem_scores_values[0] is score for A, ..., problem_scores_values[4] is score for E
    problem_scores_values = scores_input
    problem_letters = ['A', 'B', 'C', 'D', 'E']
    
    participants_data = [] # This will store tuples of (name, score)

    num_problems = 5 # There are 5 problems
    
    # Iterate through all possible non-empty combinations of problems.
    # A bitmask 'i' can represent a subset of problems.
    # 'i' ranges from 1 (binary 00001, representing only problem A)
    # to (2^num_problems - 1) (binary 11111, representing all problems A to E).
    for i in range(1, 1 << num_problems):
        current_name = ""
        current_score = 0
        
        # Check each problem (indexed 0 to num_problems-1)
        # This corresponds to problems A through E
        for j in range(num_problems):
            # Check if the j-th bit is set in the mask 'i'
            # If (i >> j) & 1 is true, it means problem_letters[j] is solved by this participant
            if ((i >> j) & 1) == 1:
                current_name += problem_letters[j]
                current_score += problem_scores_values[j]
        
        # Add the (name, score) tuple for this participant to the list
        participants_data.append((current_name, current_score))

    # Sort the list of participants.
    # The primary sort key is the score, in descending order.
    # The secondary sort key is the name, in lexicographical ascending order.
    # To achieve this, we use a lambda function that returns a tuple:
    # (-score, name). Python's default tuple comparison will achieve the desired order.
    participants_data.sort(key=lambda p: (-p[1], p[0]))

    # Print the names of the sorted participants, each on a new line.
    for p_data in participants_data:
        print(p_data[0])

if __name__ == '__main__':
    main()