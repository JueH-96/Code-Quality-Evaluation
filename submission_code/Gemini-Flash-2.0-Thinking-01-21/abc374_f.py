import collections

def solve():
    n, k, x = map(int, input().split())
    t = list(map(int, input().split()))
    
    memo = {}
    
    def get_min_dissatisfaction(order_index, last_shipment_day):
        if order_index > n:
            return 0
        if (order_index, last_shipment_day) in memo:
            return memo[(order_index, last_shipment_day)]
        
        min_dissatisfaction = float('inf')
        
        for orders_in_shipment in range(1, min(k, n - order_index + 1) + 1):
            last_order_index_in_shipment = order_index + orders_in_shipment - 1
            shipment_day = max(t[last_order_index_in_shipment-1], last_shipment_day + x)
            current_dissatisfaction = 0
            for i in range(order_index, order_index + orders_in_shipment):
                current_dissatisfaction += (shipment_day - t[i-1])
            
            remaining_dissatisfaction = get_min_dissatisfaction(order_index + orders_in_shipment, shipment_day)
            total_dissatisfaction = current_dissatisfaction + remaining_dissatisfaction
            min_dissatisfaction = min(min_dissatisfaction, total_dissatisfaction)
            
        memo[(order_index, last_shipment_day)] = min_dissatisfaction
        return min_dissatisfaction
        
    result = get_min_dissatisfaction(1, 0)
    print(result)

if __name__ == '__main__':
    solve()