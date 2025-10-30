import sys

def solve():
    n, k, x = map(int, sys.stdin.readline().split())
    t = list(map(int, sys.stdin.readline().split()))
    orders = []
    for i in range(n):
        orders.append({'id': i+1, 'placement_day': t[i], 'shipped': False})
    
    total_dissatisfaction = 0
    last_shipment_day = 0
    
    while True:
        unshipped_orders = [order for order in orders if not order['shipped']]
        if not unshipped_orders:
            break
            
        next_shipment_day = last_shipment_day + x
        if last_shipment_day == 0:
            next_shipment_day = 1
            
        ready_orders = []
        for order in unshipped_orders:
            if order['placement_day'] <= next_shipment_day:
                ready_orders.append(order)
                
        if not ready_orders:
            earliest_ready_time = min(order['placement_day'] for order in unshipped_orders)
            next_shipment_day = max(next_shipment_day, earliest_ready_time)
            ready_orders = []
            for order in unshipped_orders:
                if order['placement_day'] <= next_shipment_day:
                    ready_orders.append(order)
                    
        if not ready_orders:
            break

        ready_orders.sort(key=lambda order: order['id'])
        ship_group = ready_orders[:min(len(ready_orders), k)]
        shipment_day = next_shipment_day
        
        for order in ship_group:
            order['shipped'] = True
            total_dissatisfaction += (shipment_day - order['placement_day'])
            
        last_shipment_day = shipment_day
        
    print(total_dissatisfaction)

if __name__ == '__main__':
    solve()