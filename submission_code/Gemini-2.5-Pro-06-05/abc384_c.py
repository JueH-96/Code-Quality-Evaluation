import itertools

def solve():
    """
    Solves the programming contest problem by generating participants,
    calculating scores, and sorting them according to the rules.
    """
    
    # Read the five integer scores from a single line of input.
    scores_list = list(map(int, input().split()))
    
    # Define the problem names.
    problems = ['A', 'B', 'C', 'D', 'E']
    
    # Create a dictionary to map problem names to their scores for efficient access.
    # For example: {'A': 400, 'B': 500, 'C': 600, 'D': 700, 'E': 800}
    score_map = dict(zip(problems, scores_list))
    
    # This list will store tuples of (name, score) for each of the 31 participants.
    participants = []
    
    # Generate all non-empty subsets of problems by iterating through combination lengths 1 to 5.
    for length in range(1, len(problems) + 1):
        for combo in itertools.combinations(problems, length):
            # 'combo' is a tuple of problem names, e.g., ('B', 'C', 'E').
            # The name is formed by joining these characters.
            name = "".join(combo)
            
            # The score is the sum of scores for the problems in the combination.
            score = sum(score_map[p] for p in combo)
            
            # Store the participant's data.
            participants.append((name, score))
            
    # Sort the list of participants.
    # The key is a tuple: (-score, name).
    # - The negative score sorts scores in descending order.
    # - If scores are equal, Python's tuple comparison proceeds to the next element,
    #   sorting by name in ascending lexicographical order.
    participants.sort(key=lambda item: (-item[1], item[0]))
    
    # Print the name of each participant from the sorted list.
    for name, score in participants:
        print(name)

# Execute the solution.
solve()