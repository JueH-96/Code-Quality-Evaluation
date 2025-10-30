import sys

def solve():
    """
    Solves the election winner problem. Reads input, processes votes sequentially,
    determines the winner after each vote based on the rules (most votes, then smallest ID),
    and prints the winner's ID after each vote.
    """
    
    # Read N (number of candidates) and M (number of votes) from the first line of input.
    # sys.stdin.readline() is used for potentially faster input reading compared to input().
    N, M = map(int, sys.stdin.readline().split())
    
    # Read the sequence of M votes A from the second line of input.
    # Each element A_i is the candidate ID voted for in the i-th vote.
    # Constraint M >= 1 ensures this line exists and has M integers.
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize an array to store vote counts for each candidate.
    # Candidates are numbered 1 to N. We use an array of size N+1 for 1-based indexing.
    # Initially, all counts are 0.
    counts_array = [0] * (N + 1)
    
    # Initialize the state variables: current winner and the maximum vote count achieved.
    # According to the problem rules:
    # - The winner is the candidate with the most votes.
    # - If there's a tie for the most votes, the one with the smallest candidate number wins.
    # Initially (before any votes are counted), all candidates have 0 votes.
    # The maximum vote count is 0. All candidates are tied.
    # Among candidates 1 to N, candidate 1 has the smallest ID.
    # Therefore, candidate 1 is the initial winner with 0 votes.
    current_winner = 1
    max_votes = 0

    # Process each vote one by one, in the order they are given.
    for i in range(M):
        # Get the candidate ID for the i-th vote (using 0-based index i for A list).
        voted_candidate = A[i]
        
        # Increment the vote count for the candidate who received this vote.
        counts_array[voted_candidate] += 1
        # Get the updated vote count for this candidate.
        current_count = counts_array[voted_candidate]
        
        # Check if this vote changes the current winner based on the rules.
        
        # Case 1: The candidate who just received a vote now has strictly more votes
        # than the current maximum number of votes.
        if current_count > max_votes:
            # This candidate becomes the new winner.
            # Update the maximum vote count.
            max_votes = current_count
            # Update the current winner ID.
            current_winner = voted_candidate
        
        # Case 2: The candidate who just received a vote now has a vote count equal
        # to the current maximum number of votes. This creates or extends a tie.
        elif current_count == max_votes:
            # Apply the tie-breaking rule: the candidate with the smaller ID wins.
            # Compare the ID of the candidate who just received a vote with the current winner's ID.
            if voted_candidate < current_winner:
                # The candidate who just received a vote has a smaller ID than the current winner.
                # They become the new winner according to the tie-breaking rule.
                current_winner = voted_candidate
                # The maximum vote count (max_votes) remains unchanged.
        
        # Case 3: The candidate who just received a vote still has fewer votes
        # than the current maximum (current_count < max_votes).
        # In this case, the current winner and the maximum vote count do not change.
        # No explicit action is needed for this case.
        
        # After processing the i-th vote and updating the winner if necessary,
        # print the current winner's candidate number to standard output.
        # Each winner is printed on a new line.
        print(current_winner)

# Execute the solve function to run the program logic.
solve()