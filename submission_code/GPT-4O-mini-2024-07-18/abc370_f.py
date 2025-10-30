def can_divide(cake, K, min_weight):
    current_sum = 0
    count = 0
    for mass in cake:
        current_sum += mass
        if current_sum >= min_weight:
            count += 1
            current_sum = 0
    return count >= K

def max_min_weight(N, K, A):
    total_mass = sum(A)
    left, right = max(A), total_mass
    
    while left < right:
        mid = (left + right + 1) // 2
        if can_divide(A, K, mid):
            left = mid
        else:
            right = mid - 1
            
    return left

def count_cut_lines(N, K, A, min_weight):
    current_sum = 0
    count = 0
    cut_lines = 0
    for i in range(N):
        current_sum += A[i]
        if current_sum >= min_weight:
            count += 1
            if i < N - 1 and count < K:
                cut_lines += 1
            current_sum = 0
            
    if count == K:
        cut_lines += (N - count)
        
    return cut_lines

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    min_weight = max_min_weight(N, K, A)
    cut_lines = count_cut_lines(N, K, A, min_weight)
    
    print(min_weight, cut_lines)

if __name__ == "__main__":
    main()