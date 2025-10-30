T = int(input())
for _ in range(T):
    n, k = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    unique_A = sorted(set(A))
    min_product = float('inf')
    
    for max_val in unique_A:
        candidates = []
        for i in range(n):
            if A[i] <= max_val:
                candidates.append(B[i])
        
        if len(candidates) >= k:
            candidates.sort()
            sum_B = sum(candidates[:k])
            product = max_val * sum_B
            min_product = min(min_product, product)
    
    print(min_product)