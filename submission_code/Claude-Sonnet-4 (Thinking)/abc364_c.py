import itertools

def solve():
    N, X, Y = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    for k in range(1, N + 1):
        # Use brute force for reasonably small k
        if k <= 20:  
            for subset in itertools.combinations(range(N), k):
                sum_a = sum(A[i] for i in subset)
                sum_b = sum(B[i] for i in subset)
                
                # Check if this subset exceeds at least one threshold
                if sum_a > X or sum_b > Y:
                    # Check if we can arrange them such that first k-1 are within limits
                    for last in subset:
                        if sum_a - A[last] <= X and sum_b - B[last] <= Y:
                            return k
        else:
            # For larger k, use heuristic approach
            # Try the k dishes with highest total contribution
            dishes = list(range(N))
            dishes.sort(key=lambda i: A[i] + B[i], reverse=True)
            
            subset = dishes[:k]
            sum_a = sum(A[i] for i in subset)
            sum_b = sum(B[i] for i in subset)
            
            if sum_a > X or sum_b > Y:
                for last in subset:
                    if sum_a - A[last] <= X and sum_b - B[last] <= Y:
                        return k
    
    return N

print(solve())