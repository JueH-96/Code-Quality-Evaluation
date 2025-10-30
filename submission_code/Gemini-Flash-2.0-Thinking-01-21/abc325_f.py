import math

def solve():
    n = int(input())
    d = list(map(int, input().split()))
    l1, c1, k1 = map(int, input().split())
    l2, c2, k2 = map(int, input().split())
    
    if c1 > c2:
        l1, l2 = l2, l1
        c1, c2 = c2, c1
        k1, k2 = k2, k1
        
    memo = {}
    
    def get_min_cost(section_index, remaining_k1, remaining_k2):
        if section_index == n:
            return 0
        if (section_index, remaining_k1, remaining_k2) in memo:
            return memo[(section_index, remaining_k1, remaining_k2)]
        
        min_cost = float('inf')
        possible_for_section = False
        
        for x1 in range(remaining_k1 + 1):
            remaining_section_length = max(0, d[section_index] - x1 * l1)
            required_x2 = 0
            if remaining_section_length > 0:
                required_x2 = math.ceil(remaining_section_length / l2)
            else:
                required_x2 = 0
                
            if required_x2 <= remaining_k2:
                possible_for_section = True
                cost_for_section = x1 * c1 + required_x2 * c2
                next_cost = get_min_cost(section_index + 1, remaining_k1 - x1, remaining_k2 - required_x2)
                if next_cost != float('inf'):
                    current_cost = cost_for_section + next_cost
                    min_cost = min(min_cost, current_cost)
                    
        if not possible_for_section:
            result = float('inf')
        else:
            result = min_cost
            
        memo[(section_index, remaining_k1, remaining_k2)] = result
        return result
        
    final_cost = get_min_cost(0, k1, k2)
    if final_cost == float('inf'):
        print("-1")
    else:
        print(final_cost)

if __name__ == '__main__':
    solve()