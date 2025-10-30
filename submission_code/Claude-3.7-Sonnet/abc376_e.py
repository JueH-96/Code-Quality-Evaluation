def solve_testcase(N, K, A, B):
    # Sort the pairs (A_i, B_i) by A_i
    pairs = sorted(zip(A, B))
    
    min_value = float('inf')
    
    for i in range(K-1, N):
        A_max = pairs[i][0]
        
        # Select the K pairs with the smallest B values from the first i+1 pairs
        B_values = [pairs[j][1] for j in range(i+1)]
        B_values.sort()
        B_sum = sum(B_values[:K])
        
        # Compute the value of the expression
        value = A_max * B_sum
        min_value = min(min_value, value)
    
    return min_value

def main():
    T = int(input())
    for _ in range(T):
        N, K = map(int, input().split())
        A = list(map(int, input().split()))
        B = list(map(int, input().split()))
        print(solve_testcase(N, K, A, B))

if __name__ == "__main__":
    main()