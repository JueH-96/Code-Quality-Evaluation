# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    MOD = 998244353
    N, M = map(int, input().strip().split())
    
    # Total number of vertices
    total_vertices = N * M + 1
    
    # Sum of degrees
    sum_of_degrees = 0
    
    # Degree of vertex 0
    sum_of_degrees += N
    
    # Degree of vertices 1 to N
    sum_of_degrees += 2 * N
    
    # Degree of vertices N+1 to N*M
    sum_of_degrees += (N * M - N)
    
    # Expected number of operations
    expected_operations = sum_of_degrees * (N * M + 1)
    
    # Result modulo 998244353
    result = expected_operations % MOD
    
    print(result)

if __name__ == "__main__":
    main()