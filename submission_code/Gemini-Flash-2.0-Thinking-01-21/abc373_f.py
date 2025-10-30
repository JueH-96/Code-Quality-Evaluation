def solve():
    n, w_limit = map(int, input().split())
    items = []
    for _ in range(n):
        items.append(list(map(int, input().split())))
    
    dp = {}
    
    def get_max_happiness(item_index, remaining_weight):
        if item_index == n:
            return 0
        if (item_index, remaining_weight) in dp:
            return dp[(item_index, remaining_weight)]
        
        max_happiness = 0
        current_item_weight, current_item_value = items[item_index]
        
        for count in range(remaining_weight // current_item_weight + 1):
            weight_taken = count * current_item_weight
            happiness_from_item = count * current_item_value - count**2
            if weight_taken <= remaining_weight:
                future_happiness = get_max_happiness(item_index + 1, remaining_weight - weight_taken)
                total_happiness = happiness_from_item + future_happiness
                max_happiness = max(max_happiness, total_happiness)
                
        dp[(item_index, remaining_weight)] = max_happiness
        return max_happiness
        
    result = get_max_happiness(0, w_limit)
    print(result)

if __name__ == '__main__':
    solve()