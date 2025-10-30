import sys
import random

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    
    # A_input_list and B_input_list are 0-indexed lists.
    # A_input_list[i] stores the value A_{i+1} from the problem statement.
    # The values A_i, B_i are positive integers in the range [1, N].
    A_input_list = list(map(int, sys.stdin.readline().split()))
    B_input_list = list(map(int, sys.stdin.readline().split()))

    # rand_vals[v] stores the random hash for integer value v.
    # Since values are 1 to N, rand_vals needs N+1 size for 1-based indexing by value.
    # rand_vals[0] will be unused as element values are positive.
    rand_vals = [0] * (N + 1)
    for i in range(1, N + 1):
        rand_vals[i] = random.getrandbits(64) # Generates a random 64-bit integer

    # PA[k] will store the sum of hashes for the first k elements of A (A_1, ..., A_k).
    # PA is 1-indexed with respect to the count of elements. PA[0] = 0.
    # PA[k] = sum_{j=1 to k} rand_vals[A_j]
    # A_input_list[i] corresponds to A_{i+1} (1-based index in problem)
    PA = [0] * (N + 1)
    for i in range(N): # i iterates from 0 to N-1
        value_A_idx_plus_1 = A_input_list[i] 
        PA[i+1] = PA[i] + rand_vals[value_A_idx_plus_1]

    # PB[k] stores sum of hashes for B_1, ..., B_k. Similar logic as PA.
    PB = [0] * (N + 1)
    for i in range(N): # i iterates from 0 to N-1
        value_B_idx_plus_1 = B_input_list[i]
        PB[i+1] = PB[i] + rand_vals[value_B_idx_plus_1]
    
    # Store results in a list to print all at once at the end.
    # This can be slightly faster than printing line by line for large Q.
    output_results = []
    for _ in range(Q):
        # l_query, r_query, L_query, R_query are 1-based indices as per problem statement
        l_query, r_query, L_query, R_query = map(int, sys.stdin.readline().split())

        # Length of subsequence A[l_query...r_query] is r_query - l_query + 1
        # Length of subsequence B[L_query...R_query] is R_query - L_query + 1
        len_A_subsequence = r_query - l_query + 1
        len_B_subsequence = R_query - L_query + 1

        if len_A_subsequence != len_B_subsequence:
            output_results.append("No")
        else:
            # Problem constraints state 1 <= l_i <= r_i <= N, so min length is 1.
            # An empty subsequence would have hash 0, but this case won't occur.
            
            # Hash for A_{l_query}, ..., A_{r_query} is PA[r_query] - PA[l_query-1]
            hash_A_subsequence = PA[r_query] - PA[l_query-1]
            
            # Hash for B_{L_query}, ..., B_{R_query} is PB[R_query] - PB[L_query-1]
            hash_B_subsequence = PB[R_query] - PB[L_query-1]

            if hash_A_subsequence == hash_B_subsequence:
                output_results.append("Yes")
            else:
                output_results.append("No")
    
    sys.stdout.write("
".join(output_results) + "
")

if __name__ == '__main__':
    solve()