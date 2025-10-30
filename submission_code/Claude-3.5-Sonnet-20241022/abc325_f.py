def can_cover_section(length, l1, k1, l2, k2):
    # Try all possible combinations of sensors to cover a section
    for i in range(k1 + 1):
        for j in range(k2 + 1):
            if i * l1 + j * l2 >= length:
                return (i, j)
    return None

def solve():
    # Read input
    N = int(input())
    D = list(map(int, input().split()))
    L1, C1, K1 = map(int, input().split())
    L2, C2, K2 = map(int, input().split())
    
    # For each section, find minimum sensors needed
    total_sensors1 = 0
    total_sensors2 = 0
    
    for length in D:
        result = can_cover_section(length, L1, K1-total_sensors1, L2, K2-total_sensors2)
        if result is None:
            print(-1)
            return
        s1, s2 = result
        total_sensors1 += s1
        total_sensors2 += s2
        
        # Check if we exceeded total available sensors
        if total_sensors1 > K1 or total_sensors2 > K2:
            print(-1)
            return
    
    # Calculate total cost
    total_cost = total_sensors1 * C1 + total_sensors2 * C2
    print(total_cost)

solve()