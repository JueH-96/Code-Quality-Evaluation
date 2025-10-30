import math

def solve():
    n = int(input())
    buildings = []
    for _ in range(n):
        x_h = list(map(int, input().split()))
        buildings.append({'x': x_h[0], 'h': x_h[1]})
    
    max_critical_height = -float('inf')
    
    for i_index in range(n):
        i_building = buildings[i_index]
        j_indices = []
        for j_index in range(n):
            if i_index != j_index and 0 < buildings[j_index]['x'] < i_building['x']:
                j_indices.append(j_index)
        
        if not j_indices:
            critical_height_i = -float('inf')
        else:
            positive_heights = []
            for j_index in j_indices:
                j_building = buildings[j_index]
                numerator = i_building['h'] * j_building['x'] - i_building['x'] * j_building['h']
                denominator = j_building['x'] - i_building['x']
                if denominator == 0:
                    h_ij = float('inf') if numerator >= 0 else -float('inf')
                else:
                    h_ij = numerator / denominator
                if h_ij >= 0:
                    positive_heights.append(h_ij)
            if not positive_heights:
                critical_height_i = -float('inf')
            else:
                critical_height_i = min(positive_heights)
                
        max_critical_height = max(max_critical_height, critical_height_i)
        
    if max_critical_height < 0:
        
        all_visible_at_zero = True
        for i_index in range(n):
            i_building = buildings[i_index]
            j_indices = []
            for j_index in range(n):
                if i_index != j_index and 0 < buildings[j_index]['x'] < i_building['x']:
                    j_indices.append(j_index)
            if not j_indices:
                continue
            else:
                max_ratio = 0
                for j_index in j_indices:
                    j_building = buildings[j_index]
                    max_ratio = max(max_ratio, (i_building['x'] * j_building['h']) / j_building['x'])
                if i_building['h'] <= max_ratio:
                    all_visible_at_zero = False
                    break
        if all_visible_at_zero:
            print("-1")
        else:
            print(f"{max(0, max_critical_height):.20f}")

    else:
        print(f"{max(0, max_critical_height):.20f}")

if __name__ == '__main__':
    solve()