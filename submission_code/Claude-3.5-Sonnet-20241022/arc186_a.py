def get_row_col_sums(matrix, N):
    row_sums = [sum(matrix[i]) for i in range(N)]
    col_sums = [sum(matrix[j][i] for j in range(N)) for i in range(N)]
    return row_sums, col_sums

def can_have_k_fixed(N, K):
    # Special cases
    if K == N*N:  # All elements fixed
        return True
    if K == 0:  # No elements fixed
        if N >= 2:
            return True
        return False
        
    # For N×N matrix, if we have row sums and column sums equal,
    # the following elements must be fixed:
    # 1. Elements that must be 1: where row_sum + col_sum > N
    # 2. Elements that must be 0: where row_sum + col_sum < N
    
    # For a valid matrix with fixed elements:
    # - K must be even (except when K = N*N)
    # - K cannot be N*N - 1
    # - K cannot be 1
    if K == N*N - 1 or K == 1:
        return False
    
    # For N ≥ 2, possible values for fixed elements are:
    # 0, N*N, and even numbers between 0 and N*N that satisfy certain conditions
    if K % 2 == 1 and K != N*N:
        return False
        
    # For N ≥ 2, we can always construct matrices with even number of fixed elements
    # up to certain threshold by manipulating row and column sums
    max_possible = N*N
    min_possible = 0
    
    if K > max_possible or K < min_possible:
        return False
        
    # For N ≥ 2, we can construct matrices with even number of fixed elements
    # by having appropriate row and column sums
    if K % 2 == 0 or K == N*N:
        return True
        
    return False

def main():
    # Read input
    N, Q = map(int, input().split())
    queries = [int(input()) for _ in range(Q)]
    
    # Process each query
    for K in queries:
        if can_have_k_fixed(N, K):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()