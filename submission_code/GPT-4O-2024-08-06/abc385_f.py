# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split('
')
    
    N = int(data[0].strip())
    buildings = []
    
    for i in range(1, N + 1):
        x, h = map(int, data[i].strip().split())
        buildings.append((x, h))
    
    def can_see_all_buildings_from_height(h):
        max_slope = float('-inf')
        for i in range(N):
            x_i, h_i = buildings[i]
            slope = (h_i - h) / x_i
            if slope > max_slope:
                max_slope = slope
            else:
                return False
        return True
    
    low, high = 0.0, 1e9
    epsilon = 1e-9
    
    if can_see_all_buildings_from_height(0):
        print("-1")
        return
    
    while high - low > epsilon:
        mid = (low + high) / 2
        if can_see_all_buildings_from_height(mid):
            low = mid
        else:
            high = mid
    
    print(f"{low:.18f}")

main()