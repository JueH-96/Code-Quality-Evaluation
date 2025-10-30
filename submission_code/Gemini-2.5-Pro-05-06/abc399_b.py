# YOUR CODE HERE
import sys

def main():
    N = int(sys.stdin.readline())
    # P_scores will be a list of N integers, representing scores P_0, P_1, ..., P_{N-1}
    P_scores = list(map(int, sys.stdin.readline().split())) 

    # people_by_score[s] stores a list of original 0-based indices 
    # of people who achieved score s.
    # Scores are between 1 and 100. List size 101 for indices 0..100.
    # Index 0 of people_by_score will be unused as scores are >= 1.
    people_by_score = [[] for _ in range(101)] 

    for i in range(N): # i is the 0-based original index of a person
        score = P_scores[i]
        people_by_score[score].append(i)

    # final_ranks[k] will store the rank of person k (0-indexed based on input order).
    final_ranks = [0] * N 
    current_rank_to_assign = 1 # This is 'r' from the problem description.

    # Iterate scores from highest (100) down to lowest (1)
    for score_value in range(100, 0, -1): # score_value from 100 down to 1
        
        # Check if any person achieved this score_value
        if not people_by_score[score_value]: 
            # No one scored this value, so move to the next lower score.
            continue 
        
        # If we are here, it means people_by_score[score_value] is not empty.
        # These are the people with the current highest score among unranked.
        
        # Get list of original indices of people who achieved this score
        original_indices_for_this_score = people_by_score[score_value]
        
        # k from problem description: number of people whose score is x (current highest)
        num_people_getting_this_rank = len(original_indices_for_this_score) 
        
        # Assign current_rank_to_assign to all these people
        for original_idx in original_indices_for_this_score:
            final_ranks[original_idx] = current_rank_to_assign
        
        # Add k to r (update current_rank_to_assign for the next group)
        current_rank_to_assign += num_people_getting_this_rank
            
    # Print the rank for each person, in the original input order.
    for rank_val in final_ranks:
        sys.stdout.write(str(rank_val) + "
")

if __name__ == '__main__':
    main()