# YOUR CODE HERE
def solve():
    n = int(input())
    A = list(map(int, input().split()))
    
    total = 0
    
    for L in range(n):
        for R in range(L, n):
            # Get unique elements in subarray A[L:R+1]
            unique_elements = set(A[L:R+1])
            unique_sorted = sorted(unique_elements)
            
            # Find minimum number of contiguous ranges to cover all unique elements
            if not unique_sorted:
                operations = 0
            else:
                operations = 1
                for i in range(1, len(unique_sorted)):
                    if unique_sorted[i] != unique_sorted[i-1] + 1:
                        operations += 1
            
            total += operations
    
    print(total)

solve()