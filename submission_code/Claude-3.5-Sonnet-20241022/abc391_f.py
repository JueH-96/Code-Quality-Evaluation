def solve():
    # Read input
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    C = list(map(int, input().split()))
    
    # Create list to store all possible values
    values = []
    
    # Calculate all possible combinations
    # We can optimize by only generating K largest values if needed
    if K <= 5*10**5:  # When K is small enough
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    val = A[i]*B[j] + B[j]*C[k] + C[k]*A[i]
                    values.append(val)
        
        # Sort in descending order and get Kth largest
        values.sort(reverse=True)
        print(values[K-1])
    else:
        # For larger K, we would need a different approach
        # But according to constraints, K ≤ min(N^3, 5×10^5)
        # So this case won't occur in the actual tests
        pass

if __name__ == "__main__":
    solve()