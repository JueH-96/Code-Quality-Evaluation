def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    def can_divide_with_min_weight(min_weight):
        current_sum = 0
        count = 0
        for weight in A:
            current_sum += weight
            if current_sum >= min_weight:
                count += 1
                current_sum = 0
        return count >= K
    
    left, right = 0, sum(A) // K
    best_min_weight = 0
    
    while left <= right:
        mid = (left + right) // 2
        if can_divide_with_min_weight(mid):
            best_min_weight = mid
            left = mid + 1
        else:
            right = mid - 1
    
    # Calculate the number of uncut lines
    # If we can divide into K parts, we make exactly K-1 cuts
    uncut_lines = N - (K - 1)
    
    print(best_min_weight, uncut_lines)