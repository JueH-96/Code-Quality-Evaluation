import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())
    
    # Read N contests [L_i, R_i]
    # Store contests as a list of tuples (L_i, R_i)
    contests = []
    for _ in range(N):
        l, r = map(int, sys.stdin.readline().split())
        contests.append((l, r))

    # Read Q
    Q = int(sys.stdin.readline())
    
    # Read Q queries X
    queries = [int(sys.stdin.readline()) for _ in range(Q)]

    # Determine a safe upper bound for the final rating.
    # The maximum possible initial rating is 500000.
    # The maximum possible increase over N contests is N.
    # So, the maximum possible final rating is at most 500000 + N.
    # Given N <= 2 * 10^5, max final rating is <= 500000 + 200000 = 700000.
    # We need an exclusive upper bound for the binary search range [low, high).
    # A value of 700001 is a safe exclusive upper bound for final ratings up to 700000.
    MAX_POSSIBLE_FINAL_RATING_EXCLUSIVE_UPPER_BOUND = 500000 + 200000 + 1

    def calculate_initial_rating_from_final(final_rating):
        """
        Given a potential final rating `final_rating`, apply the inverse
        transformations contest by contest, from N down to 1, to find the
        initial rating that would lead to this final rating.

        The inverse transformation for contest i (mapping rating r_i *after*
        contest i to rating r_{i-1} *before* contest i) is:
        r_{i-1} = r_i - 1 if L_i + 1 <= r_i <= R_i + 1
        r_{i-1} = r_i otherwise

        We start with the rating *after* contest N (which is the `final_rating`),
        apply the inverse transformation for contest N (g_N) to get the rating
        *before* contest N (which is also the rating *after* contest N-1),
        then apply g_{N-1} to get the rating *before* contest N-1 (after N-2), and so on.
        After applying g_1, we get the rating *before* contest 1, which is the initial rating.
        """
        current_rating_after_contest_i = final_rating
        
        # Iterate backwards through contests from N down to 1.
        # contests list is 0-indexed, so indices are N-1 down to 0.
        for i in range(N - 1, -1, -1):
            L, R = contests[i]
            
            # The current value `current_rating_after_contest_i` represents
            # the rating *after* contest `i+1` in the perspective of the loop,
            # or equivalently, the rating *after* contest `i` according to the g_i definition.
            # We check if this rating falls in the range [L_i + 1, R_i + 1].
            # If it does, applying g_i subtracts 1.
            
            if L + 1 <= current_rating_after_contest_i <= R + 1:
                 current_rating_after_contest_i -= 1 # Apply g_i

            # After this step, `current_rating_after_contest_i` now holds the rating
            # that was present *before* contest `i`. For the next iteration (i-1),
            # this value effectively becomes the rating *after* contest `i-1`.

        # After the loop finishes (i reaches -1, meaning we processed contest 0 effectively),
        # `current_rating_after_contest_i` holds the rating that was present
        # *before* contest 1, which is the initial rating.
        initial_rating = current_rating_after_contest_i
        return initial_rating

    # Process each query
    results = []
    for X in queries:
        # For a given initial rating X, we need to find the final rating R.
        # Let F_N(X) be the final rating for initial X. We want to compute F_N(X).
        # Let G_N(R) = calculate_initial_rating_from_final(R).
        # The relationship is R = F_N(X) <=> X = G_N(R).
        # We are given X and need to find R such that G_N(R) = X.
        # The function G_N(R) is non-decreasing with R.
        # We can use binary search to find the smallest R such that G_N(R) >= X.
        # Since X is a valid initial rating, there exists an R = F_N(X) such that G_N(R) = X.
        # The smallest R satisfying G_N(R) >= X will be exactly the smallest R satisfying G_N(R) = X,
        # which is the required final rating F_N(X).

        # Binary search for the smallest R in the range [low, high)
        # such that calculate_initial_rating_from_final(R) >= X.
        low = 1 # Minimum possible final rating
        high = MAX_POSSIBLE_FINAL_RATING_EXCLUSIVE_UPPER_BOUND # Exclusive upper bound

        # The loop invariant is that the answer (the smallest R such that G_N(R) >= X)
        # is in the range [low, high).
        while low < high:
            # Calculate the middle value in the current search range.
            mid = low + (high - low) // 2
            
            # Calculate the initial rating that would result in the final rating `mid`.
            initial_val = calculate_initial_rating_from_final(mid)
            
            if initial_val < X:
                # The calculated initial rating is less than the query X.
                # This means the potential final rating `mid` is too low.
                # A larger final rating is needed to achieve an initial rating of X.
                # The answer must be in the range [mid + 1, high).
                low = mid + 1
            else: # initial_val >= X
                # The calculated initial rating is X or greater.
                # This means the potential final rating `mid` is a possible answer.
                # However, there might be a smaller final rating that also results in
                # an initial rating >= X (and possibly equals X).
                # The answer must be in the range [low, mid).
                high = mid

        # When the loop terminates, low == high. This value is the smallest
        # `mid` that satisfied the condition `calculate_initial_rating_from_final(mid) >= X`.
        # This value `low` (or `high`) is the correct final rating for initial X.
        results.append(low)

    # Print the results for all queries
    for res in results:
        print(res)

# Execute the solve function to run the program
solve()