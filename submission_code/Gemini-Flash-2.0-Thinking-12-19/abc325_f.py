import math

def solve():
    n = int(input())
    d = list(map(int, input().split()))
    l1, c1, k1 = map(int, input().split())
    l2, c2, k2 = map(int, input().split())
    
    for section_length in d:
        if section_length > k1 * l1 + k2 * l2:
            print("-1")
            return
            
    dp = {}
    
    def get_dp_value(section_index, sensors1_left, sensors2_left):
        if section_index == n:
            return 0
        if (section_index, sensors1_left, sensors2_left) in dp:
            return dp[(section_index, sensors1_left, sensors2_left)]
        
        min_cost = float('inf')
        for x1 in range(sensors1_left + 1):
            required_length = max(0, d[section_index] - x1 * l1)
            x2 = 0
            if l2 > 0:
                x2 = (required_length + l2 - 1) // l2
            else:
                if required_length > 0:
                    continue # Impossible with type-2 sensors
                    
            if x2 <= sensors2_left:
                current_cost = x1 * c1 + x2 * c2
                remaining_cost = get_dp_value(section_index + 1, sensors1_left - x1, sensors2_left - x2)
                if remaining_cost != float('inf'):
                    min_cost = min(min_cost, current_cost + remaining_cost)
                    
        dp[(section_index, sensors1_left, sensors2_left)] = min_cost
        return min_cost
        
    result_cost = get_dp_value(0, k1, k2)
    if result_cost == float('inf'):
        print("-1")
    else:
        print(result_cost)

if __name__ == '__main__':
    solve()