import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())
    # Read scores P_1 P_2 ... P_N into a list
    P = list(map(int, sys.stdin.readline().split()))

    # Initialize data structures
    # final_ranks[i] will store the rank of the person originally at index i
    final_ranks = [0] * N
    # is_determined[i] is True if person at index i has their rank determined
    is_determined = [False] * N
    
    # current_rank is the rank to be assigned to the next group of people
    current_rank = 1
    # num_determined_people counts how many people have received their rank
    num_determined_people = 0

    # Continue until all N people have their ranks determined
    while num_determined_people < N:
        # Step 1: Find the maximum score among people whose ranks are currently undetermined
        max_undetermined_score = -1 # Scores P_i are >= 1, so -1 is a safe initial minimum
        for i in range(N):
            if not is_determined[i]:
                max_undetermined_score = max(max_undetermined_score, P[i])

        # Step 2: Identify all 'k' people whose score is max_undetermined_score
        # and whose rank is still undetermined.
        people_to_rank_indices = []
        count_k = 0
        for i in range(N):
            if not is_determined[i] and P[i] == max_undetermined_score:
                people_to_rank_indices.append(i)
                count_k += 1
        
        # Step 3: Determine the rank of these 'k' people to be current_rank
        for idx in people_to_rank_indices:
            final_ranks[idx] = current_rank
            is_determined[idx] = True # Mark them as determined
        
        # Step 4: Add 'k' to current_rank for the next iteration
        current_rank += count_k
        # Update the total count of determined people
        num_determined_people += count_k

    # Print the rank of each person, one per line
    for rank in final_ranks:
        sys.stdout.write(str(rank) + '
')

# Standard boilerplate to call the solve function when the script is executed
if __name__ == '__main__':
    solve()