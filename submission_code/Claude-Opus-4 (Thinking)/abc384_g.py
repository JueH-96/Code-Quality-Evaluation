import bisect

N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
K = int(input())

for _ in range(K):
    X, Y = map(int, input().split())
    
    # Extract and sort the relevant parts of B
    B_sub = sorted(B[:Y])
    B_sum = sum(B_sub)
    
    # Compute prefix sums
    B_prefix = [0]
    for b in B_sub:
        B_prefix.append(B_prefix[-1] + b)
    
    total = 0
    for i in range(X):
        a = A[i]
        # Find the number of elements in B_sub that are <= a
        pos = bisect.bisect_right(B_sub, a)
        
        # Compute the contribution of A[i]
        total += (2 * pos - Y) * a + B_sum - 2 * B_prefix[pos]
    
    print(total)