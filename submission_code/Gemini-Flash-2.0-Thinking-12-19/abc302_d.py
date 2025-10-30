def solve():
    n, m, d = map(int, input().split())
    a_gifts = list(map(int, input().split()))
    b_gifts = list(map(int, input().split()))
    
    b_gifts.sort()
    max_sum = -1
    
    for a_gift_value in a_gifts:
        lower_bound = a_gift_value - d
        upper_bound = a_gift_value + d
        
        best_b_gift_value = -1
        
        l, r = 0, len(b_gifts) - 1
        max_valid_b_index = -1
        
        while l <= r:
            mid_index = (l + r) // 2
            if b_gifts[mid_index] <= upper_bound:
                max_valid_b_index = mid_index
                l = mid_index + 1
            else:
                r = mid_index - 1
                
        if max_valid_b_index != -1:
            candidate_b_gift_value = b_gifts[max_valid_b_index]
            if candidate_b_gift_value >= lower_bound:
                current_sum = a_gift_value + candidate_b_gift_value
                max_sum = max(max_sum, current_sum)
                
    print(max_sum)

if __name__ == '__main__':
    solve()