import sys
from bisect import bisect_left, bisect_right

def solve():
    """
    Solves the 11/22 String Subsequence problem by precomputing prefix sums
    and using binary search to find the optimal slash for each query.
    """
    # Use fast I/O
    input = sys.stdin.readline

    # Read problem size and the string
    N, Q = map(int, input().split())
    S = input().strip()

    # 1. Precomputation
    # prefix1[k] stores the count of '1's in S[0...k-1].
    # prefix2[k] stores the count of '2's in S[0...k-1].
    prefix1 = [0] * (N + 1)
    prefix2 = [0] * (N + 1)
    
    # Store indices of all '/' characters.
    slash_indices = []
    
    for i in range(N):
        prefix1[i + 1] = prefix1[i]
        prefix2[i + 1] = prefix2[i]
        if S[i] == '1':
            prefix1[i + 1] += 1
        elif S[i] == '2':
            prefix2[i + 1] += 1
        elif S[i] == '/':
            slash_indices.append(i)

    # List to store answers for fast output
    answers = []

    # 2. Process Queries
    for _ in range(Q):
        L, R = map(int, input().split())
        
        # Convert to 0-based indexing. Query range [L, R] becomes indices [l, r-1].
        l, r = L - 1, R

        # Find the slice of slash_indices that falls within the query range [l, r-1].
        start_j = bisect_left(slash_indices, l)
        end_j_exclusive = bisect_right(slash_indices, r - 1)

        # If no slashes are in the range, an 11/22 string is not possible.
        if start_j >= end_j_exclusive:
            answers.append("0")
            continue

        j_min = start_j
        j_max = end_j_exclusive - 1

        # Binary search for the "crossover" index j where num_ones >= num_twos.
        # This is equivalent to: prefix1[i] + prefix2[i+1] >= prefix1[l] + prefix2[r]
        low, high = j_min, j_max
        crossover_j = -1
        target_val = prefix1[l] + prefix2[r]

        while low <= high:
            mid_j = (low + high) // 2
            mid_i = slash_indices[mid_j]
            current_val = prefix1[mid_i] + prefix2[mid_i + 1]

            if current_val >= target_val:
                crossover_j = mid_j
                high = mid_j - 1
            else:
                low = mid_j + 1

        max_m = 0

        # The optimal slash index is at or just before the crossover point.
        candidate_j_indices = set()
        if crossover_j != -1:
            candidate_j_indices.add(crossover_j)
            if crossover_j > j_min:
                candidate_j_indices.add(crossover_j - 1)
        else:
            # No crossover means num_ones < num_twos for all slashes.
            # max of min(num_ones, num_twos) is max of num_ones.
            # num_ones is increasing, so the max is at the last slash.
            candidate_j_indices.add(j_max)
        
        for j in candidate_j_indices:
            i = slash_indices[j]
            num_ones = prefix1[i] - prefix1[l]
            num_twos = prefix2[r] - prefix2[i + 1]
            m = min(num_ones, num_twos)
            if m > max_m:
                max_m = m

        # The length of the 11/22 string is 2*m + 1.
        answers.append(str(2 * max_m + 1))

    # Print all answers at once for efficiency.
    sys.stdout.write("
".join(answers) + "
")

solve()