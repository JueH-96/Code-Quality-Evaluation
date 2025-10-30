def mountain_valley_crests(N, A):
    from bisect import bisect_right
    
    # Mapping from crease index to list of numbers on the other side of the mountain fold
    mountain_fold = {}
    
    # Calculate which numbers are on the other side of each mountain fold
    for i in range(N):
        for j in range(i+1, N):
            fold_point = A[j] - (A[j] - A[i]) * 2
            if fold_point not in mountain_fold:
                mountain_fold[fold_point] = []
            mountain_fold[fold_point].append(A[j])
    
    # We only care about these special folds
    special_folds = sorted(mountain_fold.keys())
    
    # Count the mountains on either side of each crease
    count_left = []
    count_right = [N] * (1 << 100)
    
    for i, fold in enumerate(special_folds):
        indices = mountain_fold[fold]
        left_count = sum(1 for x in indices if x < fold)
        count_left.append(left_count)
        count_right[i] = N - left_count
    
    # Binary search to find the maximum number of mountains on one side
    def max_mountains(start, end):
        max_mountains = 0
        while start < end:
            mid = start + (end - start) // 2
            i = bisect_right(special_folds, mid)
            if i:
                left_max = count_left[i-1]
                right_max = N - left_max
                max_mountains = max(max_mountains, left_max, right_max)
            if mid + 1 < len(count_right):
                mid_max = count_right[i]
                max_mountains = max(max_mountains, mid_max)
            if count_left[i-1] if i else 0 > N - (count_right[i] if i < len(count_right) else 0):
                start = mid + 1
            else:
                end = mid
        return max_mountains
    
    # Find the maximum number of mountains within the given range
    max_mountains_count = 0
    for a in A[1:-1]:
        max_mountains_count = max(max_mountains_count, max_mountains(a + 1, (1 << 100) - a - 1))
    
    return max_mountains_count

# Read input and solve
N = int(input())
A = list(map(int, input().split()))
print(mountain_valley_crests(N, A))