# YOUR CODE HERE
import sys

def solve():
    """
    Reads election data, processes votes sequentially, and determines the winner after each vote.
    """
    # Read the number of candidates (N) and votes (M).
    # Using sys.stdin.readline() for performance on large inputs.
    try:
        n, m = map(int, sys.stdin.readline().split())
        # Create an iterator for the stream of votes for memory efficiency.
        votes_stream = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Handle cases of empty or malformed input.
        return

    # 'counts' will store the vote count for each candidate.
    # Candidates are numbered 1 to N, so an array of size N+1 is used.
    counts = [0] * (n + 1)
    
    # 'max_votes' tracks the highest vote count seen so far.
    max_votes = 0
    
    # 'winner' tracks the current winner's candidate number.
    # An initial value of 0 is safe because candidate numbers start from 1.
    # The first vote will always update this to a valid candidate number.
    winner = 0

    # Collect winners in a list to print all at once at the end.
    # This is an I/O optimization for a large number of outputs.
    results = []

    for v in votes_stream:
        # Increment vote count for the candidate.
        counts[v] += 1
        
        # Check if this vote changes the winner based on the rules.
        if counts[v] > max_votes:
            # This candidate is the new undisputed winner.
            max_votes = counts[v]
            winner = v
        elif counts[v] == max_votes:
            # A tie for the most votes. Apply the tie-breaking rule (smaller candidate number).
            if v < winner:
                winner = v
        
        # Add the current winner to the results list.
        results.append(str(winner))

    # Print all the winners, each on a new line.
    print('
'.join(results))

# Run the solution
solve()