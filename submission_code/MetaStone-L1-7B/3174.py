def min_changes_beautiful(s: str) -> int:
    n = len(s)
    count0 = s.count('0')
    count1 = n - count0
    single_run_cost = min(count0, count1)
    
    min_two_run_cost = float('inf')
    
    for m in range(0, n, 2):
        if m >= n:
            break
        if m == 0:
            left_count0 = 0
        else:
            left_count0 = s[:m].count('0')
        left_count1 = m - left_count0
        cost_left = min(left_count0, left_count1)
        
        if (n - m) == 0:
            right_count0 = 0
        else:
            right_count0 = s[m:].count('0')
        right_count1 = (n - m) - right_count0
        cost_right = min(right_count0, right_count1)
        
        total = cost_left + cost_right
        if total < min_two_run_cost:
            min_two_run_cost = total
    
    total_min = min(single_run_cost, min_two_run_cost)
    return total_min

# Example usage
s = "1001"
print(min_changes_beautiful(s))  # Output: 1