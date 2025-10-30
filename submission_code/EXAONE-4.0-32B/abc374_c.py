import sys
import bisect

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    arr = list(map(int, data[1:1+n]))
    total = sum(arr)
    
    mid_index = n // 2
    left_arr = arr[:mid_index]
    right_arr = arr[mid_index:]
    
    left_subsets = []
    n_left = len(left_arr)
    for mask in range(1 << n_left):
        s = 0
        for j in range(n_left):
            if mask & (1 << j):
                s += left_arr[j]
        left_subsets.append(s)
        
    right_subsets = []
    n_right = len(right_arr)
    for mask in range(1 << n_right):
        s = 0
        for j in range(n_right):
            if mask & (1 << j):
                s += right_arr[j]
        right_subsets.append(s)
        
    right_subsets.sort()
    
    def check(M):
        for L in left_subsets:
            low_bound = total - M - L
            high_bound = M - L
            if high_bound < 0:
                continue
            actual_low = max(0, low_bound)
            if actual_low > high_bound:
                continue
            pos = bisect.bisect_left(right_subsets, actual_low)
            if pos < len(right_subsets) and right_subsets[pos] <= high_bound:
                return True
        return False
    
    low_val = (total + 1) // 2
    high_val = total
    ans = high_val
    while low_val <= high_val:
        mid_val = (low_val + high_val) // 2
        if check(mid_val):
            ans = mid_val
            high_val = mid_val - 1
        else:
            low_val = mid_val + 1
            
    print(ans)

if __name__ == "__main__":
    main()