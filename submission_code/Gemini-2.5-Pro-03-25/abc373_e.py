# YOUR CODE HERE
import sys
import bisect

def solve():
    # Read input values for N, M, K
    N, M, K = map(int, sys.stdin.readline().split())
    # Read initial votes for each candidate
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate the total votes already counted
    current_sum_A = sum(A)
    # Calculate the number of remaining votes
    R = K - current_sum_A 

    # Store pairs of (score, original_index) to facilitate sorting while keeping track of original indices
    B = []
    for idx in range(N):
        B.append((A[idx], idx))

    # Sort candidates based on their initial scores in descending order
    B.sort(key=lambda x: x[0], reverse=True)

    # Extract the sorted scores into a list
    sorted_A = [x[0] for x in B]
    
    # Create a mapping from original candidate index to their rank (0-based) after sorting
    # rank[original_idx] = rank
    rank = [-1] * N
    for r in range(N):
        # B[r] is a tuple (score, original_idx) for the candidate with rank r
        original_idx = B[r][1]
        rank[original_idx] = r

    # Calculate prefix sums of the sorted scores.
    # prefix_sum[k] stores the sum of scores of candidates with ranks 0 to k-1.
    prefix_sum = [0] * (N + 1)
    for k in range(N):
        prefix_sum[k+1] = prefix_sum[k] + sorted_A[k]

    # Create a version of the scores sorted in ascending order.
    # This is useful for binary search operations using the `bisect` module.
    A_asc = sorted_A[::-1] 

    # List to store the results (minimum votes needed for each candidate)
    results = []

    # Iterate through each candidate (using their original index i)
    for i in range(N): 
        
        current_A_i = A[i] # Initial score of candidate i
        current_rank_i = rank[i] # Rank of candidate i (0-based)

        # Binary search for the minimum additional votes X needed for candidate i to guarantee victory.
        # Search space for X is [0, R].
        low = 0
        high = R
        # Initialize min_X to a value outside the valid range [0, R].
        # If it remains this value after the search, it means no solution was found.
        min_X = R + 1 

        # Standard binary search loop
        while low <= high:
            # Calculate midpoint X value
            mid_X = (low + high) // 2
            
            # Calculate the target score T. If candidate i gets mid_X votes, their score is A[i] + mid_X.
            # For guarantee condition, we consider opponents needing strictly more votes. 
            # Effectively, they need to reach score A[i] + mid_X + 1.
            target_score_T = current_A_i + mid_X + 1
            
            # Find p: the count of candidates whose current score is >= target_score_T.
            # Use `bisect_left` on the ascending sorted list `A_asc`.
            # `bisect_left(A_asc, T)` finds the insertion point `ins_idx` for T.
            # All elements `A_asc[ins_idx:]` are >= T.
            # The count of such elements is N - ins_idx. This is p.
            ins_idx = bisect.bisect_left(A_asc, target_score_T)
            p = N - ins_idx 
            
            # Calculate f(X) = sum(max(0, T - A_k)) for the M 'most dangerous' opponents.
            # The definition of 'most dangerous' depends on whether candidate i is initially among top M.
            f_X = 0 
            
            # The logic uses 0-based indexing for ranks and prefix sums.
            # prefix_sum[k] = sum_{l=0}^{k-1} sorted_A[l]
            # Sum_{l=a}^{b} sorted_A[l] = prefix_sum[b+1] - prefix_sum[a]
            
            if current_rank_i >= M: # Case 1: Candidate i is NOT initially among the top M candidates (ranks 0..M-1)
                # The M most dangerous opponents are the top M candidates (ranks 0 to M-1).
                
                # We need sum (T - sorted_A[k]) for k in {0..M-1} where sorted_A[k] < T.
                # sorted_A[k] < T is equivalent to rank k >= p.
                # So, we sum over k in {0..M-1} AND k >= p. This range is {p..M-1}.
                
                if p < M: # Check if there are any opponents in this group with score < T
                    # The sum needed is sum(sorted_A[k] for k=p..M-1)
                    # This sum is prefix_sum[M] - prefix_sum[p]
                    sum_A_k = prefix_sum[M] - prefix_sum[p]
                    # f(X) = count * T - sum = (M-p) * T - sum_A_k
                    f_X = (M - p) * target_score_T - sum_A_k
                # else p >= M: All top M opponents already have score >= T. f(X) is 0.

            else: # Case 2: Candidate i IS initially among the top M candidates (rank r = current_rank_i is in 0..M-1)
                # The M most dangerous opponents are candidates with ranks {0..M} \ {r}. Let this set be K.
                
                # We need sum (T - sorted_A[k]) for k in K where sorted_A[k] < T.
                # sorted_A[k] < T is equivalent to rank k >= p.
                # So, we sum over k in K AND k >= p.
                
                if p > M: 
                    # Target score T is relatively low. All candidates in K have score >= T. f(X) is 0.
                    f_X = 0
                elif p <= current_rank_i: # p <= r (rank of candidate i)
                    # Sum over k in {p..r-1} U {r+1..M}.
                    # Count of terms = (r - p) + (M - r) = M - p.
                    count = M - p
                    # Sum of scores A_k = sum(k=p..r-1) + sum(k=r+1..M)
                    # Using prefix sums: (prefix_sum[r] - prefix_sum[p]) + (prefix_sum[M+1] - prefix_sum[r+1])
                    sum_A_k = (prefix_sum[current_rank_i] - prefix_sum[p]) + \
                              (prefix_sum[M+1] - prefix_sum[current_rank_i+1])
                    # f(X) = count * T - sum_A_k
                    f_X = count * target_score_T - sum_A_k
                else: # current_rank_i < p <= M  (i.e., r < p <= M)
                    # Sum over k in {p..M}. Note r < p, so r is not included in this range.
                    # Count of terms = M - p + 1.
                    count = M - p + 1
                    # Sum of scores A_k = sum(k=p..M) = prefix_sum[M+1] - prefix_sum[p]
                    sum_A_k = prefix_sum[M+1] - prefix_sum[p]
                    # f(X) = count * T - sum_A_k
                    f_X = count * target_score_T - sum_A_k

            # Calculate g(X) = f(X) + X. This represents the total 'cost' related to candidate i receiving X votes.
            g_X = f_X + mid_X
            
            # Check the guarantee condition: g(X) > R.
            # If true, it means that the minimum votes needed by opponents to foil candidate i's win
            # plus the votes X given to candidate i exceeds the total remaining votes R.
            # This guarantees candidate i's win.
            if g_X > R:
                # If condition holds, mid_X might be the minimum X. Store it and try smaller X.
                min_X = mid_X 
                high = mid_X - 1
            else:
                # If condition fails, candidate i needs more votes. Try larger X.
                low = mid_X + 1 

        # After binary search:
        # min_X contains the minimum votes needed if a solution was found within [0, R].
        # Otherwise, min_X is still R+1.
        if min_X > R:
            # If min_X > R, it means even giving all R remaining votes to candidate i is not enough to guarantee victory.
            results.append("-1") 
        else:
             # Otherwise, min_X is the minimum required additional votes.
             results.append(str(min_X))

    # Print the results for all candidates separated by spaces.
    print(*(results))

# Execute the solve function
solve()