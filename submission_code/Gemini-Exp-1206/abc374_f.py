def solve():
    n, k, x = map(int, input().split())
    t = list(map(int, input().split()))

    dp = {}

    def get_dp(idx, last_shipment_day):
        if idx == n:
            return 0

        if (idx, last_shipment_day) in dp:
            return dp[(idx, last_shipment_day)]

        ans = float('inf')
        for num_to_ship in range(1, min(k, n - idx) + 1):
            shipment_day = max(t[idx + num_to_ship - 1], last_shipment_day)
            
            current_dissatisfaction = 0
            for i in range(num_to_ship):
                current_dissatisfaction += (shipment_day - t[idx + i])

            ans = min(ans, current_dissatisfaction + get_dp(idx + num_to_ship, shipment_day + x))

        dp[(idx, last_shipment_day)] = ans
        return ans

    print(get_dp(0, 1))

solve()