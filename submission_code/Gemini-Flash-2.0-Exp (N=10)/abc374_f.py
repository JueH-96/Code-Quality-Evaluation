def solve():
    n, k, x = map(int, input().split())
    t = list(map(int, input().split()))

    dp = {}

    def calculate_dissatisfaction(current_index, last_ship_day):
        if current_index == n:
            return 0

        if (current_index, last_ship_day) in dp:
            return dp[(current_index, last_ship_day)]

        min_dissatisfaction = float('inf')

        # Option 1: Don't ship any orders now and move to next index
        min_dissatisfaction = min(min_dissatisfaction, calculate_dissatisfaction(current_index + 1, last_ship_day))

        # Option 2: Ship some orders now
        for num_orders_to_ship in range(1, min(k, n - current_index) + 1):
            
            ship_day = max(last_ship_day + x, t[current_index])
            
            current_dissatisfaction = 0
            for i in range(current_index, current_index + num_orders_to_ship):
                current_dissatisfaction += (ship_day - t[i])
            
            min_dissatisfaction = min(min_dissatisfaction, current_dissatisfaction + calculate_dissatisfaction(current_index + num_orders_to_ship, ship_day))
        
        dp[(current_index, last_ship_day)] = min_dissatisfaction
        return min_dissatisfaction

    print(calculate_dissatisfaction(0, 0))

solve()