import sys

# The solve function encapsulates the main logic
def solve():
    # Read N (number of candidates) and M (number of votes)
    # from the first line of input.
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    M = int(line1[1])

    # Read the list of votes A from the second line of input.
    # A[i] is the candidate number for the (i+1)-th vote.
    line2 = sys.stdin.readline().split()
    A = [int(x) for x in line2]

    # Initialize a list to store vote counts for each candidate.
    # counts[i] will store the number of votes for candidate i.
    # We use a list of size N+1 to easily use 1-based indexing for candidates (1 to N).
    # Index 0 will be unused.
    counts = [0] * (N + 1)

    # Initialize variables to keep track of the current winner and the maximum vote count.
    # The problem asks for the winner after the first i votes, for i = 1 to M.
    # The state before any votes (0 votes) can be considered as a base case for comparison.
    # After 0 votes, everyone has 0 votes. According to the tie-breaking rule,
    # candidate 1 wins the tie among those with 0 votes.
    # So, we initialize max_votes to 0 and current_winner to 1.
    max_votes = 0
    current_winner = 1

    # Process the votes one by one in the given order.
    # The loop iterates M times, corresponding to processing the 1st vote, 2nd vote, ..., M-th vote.
    for i in range(M):
        # Get the candidate who received the (i+1)-th vote (which is A[i] in 0-based indexing)
        candidate = A[i]

        # Increment the vote count for this candidate
        counts[candidate] += 1
        new_count = counts[candidate]

        # Check if this vote changes the winner based on the new vote counts.

        # Case 1: The current candidate now has strictly more votes than the previous maximum.
        # This candidate becomes the new winner.
        if new_count > max_votes:
            max_votes = new_count
            current_winner = candidate
        # Case 2: The current candidate is tied with the current maximum vote count.
        # We apply the tie-breaking rule: the candidate with the smaller number wins.
        # If the current candidate's number is smaller than the current winner's number,
        # the current candidate becomes the new winner.
        elif new_count == max_votes:
            if candidate < current_winner:
                current_winner = candidate
        # Case 3: The current candidate's vote count is still less than the maximum.
        # The winner does not change in this case.

        # After processing the (i+1)-th vote, we have counted the first i+1 votes.
        # Print the winner based on these counts.
        print(current_winner)

# Call the solve function to run the program
solve()