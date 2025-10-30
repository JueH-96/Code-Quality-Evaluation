import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    A_votes = list(map(int, sys.stdin.readline().split()))

    # counts[k] stores the number of votes for candidate k.
    # Candidates are numbered 1 to N. We use 1-based indexing for `counts` array.
    # `counts[0]` will be unused.
    counts = [0] * (N + 1)
    
    # current_max_votes stores the highest number of votes any candidate has received so far.
    # Initialized to 0, as initially no votes are cast.
    current_max_votes = 0
    
    # current_winner stores the candidate number of the current winner.
    # According to the rules: if multiple candidates have the most votes,
    # the one with the smallest candidate number is the winner.
    # Initially, all candidates have 0 votes. Candidate 1 has the smallest number among them.
    # So, candidate 1 is the winner by default before any votes are counted.
    current_winner = 1 

    # Iterate through each vote in the order they are cast
    for i in range(M):
        voted_candidate = A_votes[i] # The candidate receiving the current vote
        
        # Increment the vote count for this candidate
        counts[voted_candidate] += 1
        
        # Get the new total vote count for the candidate who just received a vote
        candidate_new_vote_count = counts[voted_candidate]
        
        # Now, determine if the winner needs to be updated based on this new vote.
        
        # Case 1: The candidate who just received a vote now has strictly more votes
        # than the previous `current_max_votes`.
        # This candidate becomes the new winner, regardless of their candidate number,
        # because they are the sole leader in vote count.
        if candidate_new_vote_count > current_max_votes:
            current_max_votes = candidate_new_vote_count
            current_winner = voted_candidate
        # Case 2: The candidate who just received a vote now has a vote count
        # equal to the `current_max_votes`.
        # This means they are tied with the previous winner (and possibly others)
        # in terms of vote count. According to the tie-breaking rule, the candidate
        # with the smaller number wins. So, if this `voted_candidate` has a
        # smaller number than the `current_winner`, they become the new `current_winner`.
        elif candidate_new_vote_count == current_max_votes:
            if voted_candidate < current_winner:
                current_winner = voted_candidate
                # `current_max_votes` remains the same in this sub-case, as the new
                # winner has the same (maximum) number of votes.
        # Case 3: The candidate who just received a vote still has fewer votes
        # than `current_max_votes`. (i.e. candidate_new_vote_count < current_max_votes)
        # In this situation, they cannot be the winner. The `current_winner` and
        # `current_max_votes` remain unchanged. (No specific action needed in code).
        
        # After processing the vote and updating the winner if necessary,
        # print the candidate number of the current winner.
        sys.stdout.write(str(current_winner) + "
")

if __name__ == '__main__':
    main()