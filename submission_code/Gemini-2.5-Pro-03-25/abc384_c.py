import sys

# Function to solve the problem
def solve():
    # Read the scores for problems A, B, C, D, E from standard input
    # The input is expected to be a single line of 5 space-separated integers.
    scores_list = list(map(int, sys.stdin.readline().split()))
    
    # Create a mapping from problem index (0-4) to score for easier lookup.
    # Index 0 corresponds to problem A, 1 to B, ..., 4 to E.
    problem_scores = {
        0: scores_list[0], # Score for A
        1: scores_list[1], # Score for B
        2: scores_list[2], # Score for C
        3: scores_list[3], # Score for D
        4: scores_list[4]  # Score for E
    }
    
    # Create a list of problem characters corresponding to indices 0-4.
    problem_chars = ['A', 'B', 'C', 'D', 'E']
    
    # List to store participant data. Each element will be a tuple (score, name).
    participants = [] 
    
    # There are 5 problems, leading to 2^5 = 32 possible subsets of problems.
    # The problem states participants correspond to non-empty subsequences,
    # which means non-empty subsets of problems.
    # We iterate through integers from 1 to 31 (inclusive). 
    # The binary representation of each integer `i` represents a unique non-empty subset.
    # A '1' at the k-th bit position means the k-th problem is included in the subset.
    for i in range(1, 1 << 5): # 1 << 5 is 32. range(1, 32) iterates from 1 to 31.
        current_score = 0
        current_name = ""
        
        # Iterate through each problem index k from 0 to 4 (representing A to E).
        for k in range(5):
            # Check if the k-th bit of i is set (equal to 1).
            # The expression (i >> k) & 1 isolates the k-th bit (0-indexed from the right).
            if (i >> k) & 1:
                # If the k-th bit is set, this means the participant solved the k-th problem.
                # Add the score of the k-th problem to the participant's total score.
                current_score += problem_scores[k]
                # Append the character corresponding to the k-th problem to the participant's name.
                # The problems are always added in the order A, B, C, D, E based on index k.
                current_name += problem_chars[k]
        
        # Store the calculated total score and the constructed name as a tuple in the participants list.
        participants.append((current_score, current_name))

    # Sort the participants list based on the specified criteria:
    # 1. Primary sorting key: score, in descending order.
    # 2. Secondary sorting key: name, in lexicographical ascending order (dictionary order).
    # We use a lambda function to define a custom sort key tuple: (-score, name).
    # Negating the score achieves descending order for scores because Python sorts tuples element by element in ascending order.
    # The name is used directly for the secondary key, ensuring ascending lexicographical order.
    participants.sort(key=lambda p: (-p[0], p[1]))
    
    # Print the names of the participants in the sorted order.
    # Each name should be printed on a new line to match the required output format.
    for p in participants:
        # Each element `p` in the sorted list is a tuple (score, name).
        # We need to print the second element, which is the name `p[1]`.
        print(p[1])

# Call the solve function to execute the program logic.
solve()