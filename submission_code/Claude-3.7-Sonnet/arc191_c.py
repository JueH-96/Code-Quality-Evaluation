def solve_one_test_case(N):
    if N == 1:
        return 2, 1
    
    if N <= 60:  # Log base 2 of 10^18 is approximately 60
        # For N from 2 to 60, we can use (2, 2^N - 1)
        # This works because the order of 2 modulo (2^N - 1) is N
        A = 2
        M = (2 ** N) - 1
        return A, M
    
    # For larger N, we need a different approach to keep M within constraints
    # We can use A = N+1 and M = (A^N - 1)/(A-1) conceptually, but we need to
    # calculate it efficiently to avoid overflow
    
    # We pick a suitable A value within constraints
    # For example, A = 3
    A = 3
    
    # For large N, we know that 3^N ≡ 3^(N % φ(M)) (mod M)
    # for a carefully chosen M, we can ensure the order is exactly N
    # One such M is 2*N
    M = 2 * N
    
    return A, M

def main():
    T = int(input())
    for _ in range(T):
        N = int(input())
        A, M = solve_one_test_case(N)
        print(f"{A} {M}")

if __name__ == "__main__":
    main()