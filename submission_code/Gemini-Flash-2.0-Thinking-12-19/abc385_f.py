import sys

def solve():
    n = int(sys.stdin.readline())
    buildings = []
    for _ in range(n):
        x_h = list(map(int, sys.stdin.readline().split()))
        buildings.append({'x': x_h[0], 'h': x_h[1]})
    
    def can_see_all_at_height_zero():
        for j_index in range(1, n):
            max_l_ij_zero = 0
            found_blocker = False
            for i_index in range(j_index):
                l_ij_zero = (buildings[j_index]['x'] * buildings[i_index]['h']) / buildings[i_index]['x']
                max_l_ij_zero = max(max_l_ij_zero, l_ij_zero)
            if buildings[j_index]['h'] <= max_l_ij_zero:
                return False
        return True
        
    if can_see_all_at_height_zero():
        print("-1")
        return
        
    candidate_heights = [0]
    for i_index in range(n):
        for j_index in range(i_index + 1, n):
            building_i = buildings[i_index]
            building_j = buildings[j_index]
            ratio_i = building_i['h'] / building_i['x']
            ratio_j = building_j['h'] / building_j['x']
            if ratio_i >= ratio_j:
                h_ij_numerator = building_j['x'] * building_i['h'] - building_i['x'] * building_j['h']
                h_ij_denominator = building_j['x'] - building_i['x']
                if h_ij_denominator == 0:
                    continue
                h_ij = h_ij_numerator / h_ij_denominator
                if h_ij >= 0:
                    candidate_heights.append(h_ij)
                    
    max_height = max(candidate_heights)
    print(f"{max_height:.20f}")

if __name__ == '__main__':
    solve()