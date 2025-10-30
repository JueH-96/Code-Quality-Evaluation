import sys

# It's good practice to use fast I/O
input = sys.stdin.readline

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    K_queries = int(input())
    
    queries_params = []
    for i in range(K_queries):
        X, Y = map(int, input().split())
        # Store original 1-indexed counts for formula, and 0-indexed limits for iteration
        queries_params.append({'X_count': X, 'Y_count': Y, 
                               'X_limit': X - 1, 'Y_limit': Y - 1, 'id': i})

    # Precompute prefix sums for A and B
    # ps_A[k] stores sum of A_1...A_k (1-indexed)
    ps_A = [0] * (N + 1)
    for i in range(N):
        ps_A[i+1] = ps_A[i] + A[i]
    
    ps_B = [0] * (N + 1)
    for i in range(N):
        ps_B[i+1] = ps_B[i] + B[i]

    results_output = ["" for _ in range(K_queries)]

    # This is the O(K * X * Y) part, which is too slow for large N, K.
    # It's included to show the formula logic.
    # A faster data structure is needed for S_plus.
    for q_obj in queries_params:
        X_count, Y_count = q_obj['X_count'], q_obj['Y_count']
        X_limit, Y_limit = q_obj['X_limit'], q_obj['Y_limit'] # 0-indexed iteration limits
        qid = q_obj['id']
        
        current_S_plus = 0
        # Calculate S^+(X,Y) = sum_{i=0 to X_limit} sum_{j=0 to Y_limit, A[i] >= B[j]} (A[i] - B[j])
        if X_limit >= 0 and Y_limit >= 0: # Query involves elements
            for i in range(X_limit + 1): # Iterate A_0...A_{X_limit}
                val_A_i = A[i]
                
                # Inner part for S_plus for fixed A_i:
                # A_i * (count B_j s.t. j <= Y_limit, B_j <= A_i) - (sum B_j s.t. j <= Y_limit, B_j <= A_i)
                
                # This can be done by iterating B_j for j <= Y_limit
                count_Bj_le_Ai = 0
                sum_Bj_le_Ai = 0
                for j_idx in range(Y_limit + 1): # Iterate B_0...B_{Y_limit}
                    val_B_j = B[j_idx]
                    if val_B_j <= val_A_i:
                        count_Bj_le_Ai += 1
                        sum_Bj_le_Ai += val_B_j
                current_S_plus += val_A_i * count_Bj_le_Ai - sum_Bj_le_Ai
        
        # Final sum: 2 * S_plus - (Y_count * PS_A[X_count] - X_count * PS_B[Y_count])
        term_Y_PSA = Y_count * ps_A[X_count]
        term_X_PSB = X_count * ps_B[Y_count]
        term_2_Splus = 2 * current_S_plus
        
        final_ans = term_2_Splus - (term_Y_PSA - term_X_PSB)
        results_output[qid] = str(final_ans)

    sys.stdout.write("
".join(results_output) + "
")

solve()