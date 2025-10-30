import sys

def run():
    """
    Reads input, solves the problem using dynamic programming, and prints the answer.
    """
    try:
        # Read problem inputs from standard input
        N_str = sys.stdin.readline()
        if not N_str:
            return
        N = int(N_str)
        A = list(map(int, sys.stdin.readline().split()))
        S = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Gracefully handle empty input or malformed lines
        return

    # DP states:
    # count_m[v_m]: Number of 'M's with value v_m encountered so far.
    count_m = [0] * 3
    # count_me[v_m][v_e]: Number of 'ME' subsequences with values (v_m, v_e)
    # where the 'M' appeared before the 'E'.
    count_me = [[0] * 3 for _ in range(3)]

    total_sum = 0

    # Precompute a table for mex({v0, v1, v2}) to avoid re-computation.
    # This is a constant time operation.
    mex_table = [[[0] * 3 for _ in range(3)] for _ in range(3)]
    for v0 in range(3):
        for v1 in range(3):
            for v2 in range(3):
                s = {v0, v1, v2}
                m = 0
                while m in s:
                    m += 1
                mex_table[v0][v1][v2] = m

    # Iterate through the sequence, updating DP states and calculating the sum.
    for i in range(N):
        val = A[i]
        char = S[i]
        
        if char == 'M':
            # Found an 'M'. Increment the count for its associated value.
            count_m[val] += 1
        elif char == 'E':
            # Found an 'E'. It can form 'ME' pairs with any preceding 'M'.
            # For each possible value v_m of a preceding 'M', add the number of such 'M's
            # to the count of 'ME' pairs with values (v_m, val).
            for v_m in range(3):
                count_me[v_m][val] += count_m[v_m]
        elif char == 'X':
            # Found an 'X'. It can form 'MEX' triplets with any preceding 'ME' pair.
            # For each possible value pair (v_m, v_e) of a preceding 'ME' pair,
            # calculate its contribution and add to the total sum.
            for v_m in range(3):
                for v_e in range(3):
                    num_pairs = count_me[v_m][v_e]
                    if num_pairs > 0:
                        # Get the precomputed mex value for the triplet {v_m, v_e, val}.
                        m = mex_table[v_m][v_e][val]
                        # Add the total contribution from these pairs to the sum.
                        total_sum += num_pairs * m
                        
    print(total_sum)

# Execute the main logic
run()