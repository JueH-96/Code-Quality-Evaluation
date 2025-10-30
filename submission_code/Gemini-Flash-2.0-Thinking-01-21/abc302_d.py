def solve():
    n, m, d = map(int, input().split())
    a_gifts = list(map(int, input().split()))
    b_gifts = list(map(int, input().split()))
    
    b_gifts.sort()
    max_sum = -1
    
    for a_gift in a_gifts:
        lower_bound = a_gift - d
        upper_bound = a_gift + d
        
        start_index = -1
        low = 0
        high = len(b_gifts) - 1
        first_valid_index = -1
        while low <= high:
            mid = (low + high) // 2
            if b_gifts[mid] >= lower_bound:
                first_valid_index = mid
                high = mid - 1
            else:
                low = mid + 1
                
        last_index = -1
        low = 0
        high = len(b_gifts) - 1
        last_valid_index = -1
        while low <= high:
            mid = (low + high) // 2
            if b_gifts[mid] <= upper_bound:
                last_valid_index = mid
                low = mid + 1
            else:
                high = mid - 1
                
        if first_valid_index != -1 and last_valid_index != -1 and first_valid_index <= last_valid_index:
            best_b_gift = b_gifts[last_valid_index]
            current_sum = a_gift + best_b_gift
            if current_sum > max_sum:
                max_sum = current_sum
                
    print(max_sum)

if __name__ == '__main__':
    solve()