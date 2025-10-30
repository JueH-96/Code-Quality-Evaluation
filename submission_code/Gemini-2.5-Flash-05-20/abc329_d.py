import sys

def solve():
    # Read N and M from the first line of input
    N, M = map(int, sys.stdin.readline().split())

    # Read the list of votes A from the second line
    # A_i are 1-indexed, so we can directly use them as array indices.
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize a list to store vote counts for each candidate.
    # counts[k] will store votes for candidate k.
    # We use N+1 size to support 1-based indexing for candidates (from 1 to N).
    # counts[0] will remain unused.
    counts = [0] * (N + 1)

    # Initialize the current winner and their maximum vote count.
    # According to the rules:
    # 1. The candidate with the most votes wins.
    # 2. If tied, the one with the smallest candidate number wins.
    #
    # If no votes have been cast (or all candidates have 0 votes),
    # candidate 1 is the winner by the tie-breaking rule.
    current_winner_candidate = 1
    current_max_votes = 0

    # Process each vote in the order it was cast
    for i in range(M):
        voted_candidate = A[i]
        
        # Increment the vote count for the candidate who just received a vote
        counts[voted_candidate] += 1

        # Determine if the winner needs to be updated based on the new vote count
        
        # Scenario 1: The candidate who just received a vote now has strictly MORE votes
        # than the current leading candidate. This candidate becomes the new clear winner.
        if counts[voted_candidate] > current_max_votes:
            current_max_votes = counts[voted_candidate]
            current_winner_candidate = voted_candidate
        
        # Scenario 2: The candidate who just received a vote is now TIED with the
        # current maximum vote count. In this case, we apply the tie-breaking rule.
        # If the newly voted candidate has a smaller candidate number than the current winner,
        # they become the new winner. The max votes count remains the same.
        elif counts[voted_candidate] == current_max_votes:
            if voted_candidate < current_winner_candidate:
                current_winner_candidate = voted_candidate
        
        # Scenario 3: The candidate who just received a vote still has FEWER votes
        # than the current maximum. The current winner's lead is not affected by this vote,
        # so the winner remains unchanged.

        # Print the current winner's candidate number to standard output, followed by a newline.
        sys.stdout.write(str(current_winner_candidate) + '
')

# Call the solve function to execute the program logic
solve()