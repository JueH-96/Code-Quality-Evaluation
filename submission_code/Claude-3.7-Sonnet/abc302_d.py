from bisect import bisect_right

def max_gift_sum(A, B, D):
    B.sort()  # Sort array B to use binary search
    max_sum = -1
    
    for a in A:
        # Find the index of the largest b such that b <= a + D
        idx = bisect_right(B, a + D) - 1
        
        # Check if there's a valid b (i.e., b >= a - D)
        if idx >= 0 and B[idx] >= a - D:
            max_sum = max(max_sum, a + B[idx])
    
    return max_sum

# Read input
N, M, D = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

print(max_gift_sum(A, B, D))