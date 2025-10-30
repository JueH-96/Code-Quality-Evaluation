def longest_increasing_subsequence(arr):
    if not arr:
        return 0
    
    lis = []
    
    for num in arr:
        # Binary search to find the position to insert or replace
        left, right = 0, len(lis) - 1
        idx = len(lis)  # Default: append to the end
        
        while left <= right:
            mid = (left + right) // 2
            if lis[mid] < num:
                left = mid + 1
            else:
                idx = mid
                right = mid - 1
        
        # Insert or replace the number at the appropriate position
        if idx == len(lis):
            lis.append(num)
        else:
            lis[idx] = num
    
    return len(lis)

def solve_query(A, R, X):
    # Filter elements up to R-th position (1-based indexing)
    filtered_A = [A[i] for i in range(R) if A[i] <= X]
    return longest_increasing_subsequence(filtered_A)

# Read the input
N, Q = map(int, input().split())
A = list(map(int, input().split()))

for _ in range(Q):
    R, X = map(int, input().split())
    print(solve_query(A, R, X))