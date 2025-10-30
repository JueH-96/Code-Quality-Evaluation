import bisect

def main():
    import sys
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    def is_possible(x):
        total = sum(A)
        if total < x * K:
            return False
        current_sum = 0
        segments = 0
        found_first = False
        first_segment_sum = 0
        max_rest = 0
        merged_possible = False
        
        for num in A:
            current_sum += num
            if current_sum >= x:
                if not found_first:
                    found_first = True
                    segments = 1
                    first_segment_sum = current_sum
                    current_sum = 0
                else:
                    segments += 1
                    current_sum = 0
        
        if current_sum == 0:
            return segments <= K
        
        merged_sum = first_segment_sum + current_sum
        if merged_sum >= x:
            segments -= 1  # Because we merge the first and last segments
            return segments <= K
        
        return False
    
    # Binary search to find x
    low = max(A)
    high = sum(A)
    x = low
    while low <= high:
        mid = (low + high) // 2
        if is_possible(mid):
            x = mid
            low = mid + 1
        else:
            high = mid - 1
    
    # Compute y
    y = 0
    for i in range(N):
        if A[i] + A[(i+1) % N] < x:
            y +=1
    
    print(x, y)

if __name__ == "__main__":
    main()