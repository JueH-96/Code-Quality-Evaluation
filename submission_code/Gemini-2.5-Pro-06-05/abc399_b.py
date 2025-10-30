import sys

def solve():
    """
    Reads the number of participants and their scores, calculates their ranks
    based on the specified rules, and prints the rank for each participant.
    """
    try:
        # Read the number of people, N.
        n = int(sys.stdin.readline())
        # Read the scores of the N people.
        p = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # Handles cases with empty input lines, though not expected by problem constraints.
        return

    # Create a list of tuples: (original_index, score).
    # This allows us to sort by score while keeping track of the original person.
    indexed_p = list(enumerate(p))

    # Sort the list of pairs by score in descending order.
    # The `key=lambda x: x[1]` tells `sorted` to use the score (the second item
    # in the tuple) for comparison. `reverse=True` ensures high scores come first.
    sorted_p = sorted(indexed_p, key=lambda x: x[1], reverse=True)

    # Initialize a list to store the final rank of each person. The index
    # of this list corresponds to the person's original 0-based index.
    ranks = [0] * n
    
    # This variable will hold the rank to be assigned.
    current_rank = 0

    # Iterate through the sorted list to assign ranks.
    for i in range(n):
        # Get the original index of the person at position `i` in the sorted list.
        original_index = sorted_p[i][0]

        # The rank is updated only when we encounter a new, lower score.
        # The `or` short-circuits, so `sorted_p[i-1]` is not accessed for `i=0`.
        if i == 0 or sorted_p[i][1] < sorted_p[i - 1][1]:
            # The new rank is the person's 1-based position in the sorted list.
            current_rank = i + 1
        
        # Assign the calculated rank. If scores are tied, `current_rank` remains
        # unchanged from the previous iteration, correctly assigning the same rank.
        ranks[original_index] = current_rank

    # Print the rank for each person, from person 1 to N, on a new line.
    for rank in ranks:
        print(rank)

# Run the solution
solve()