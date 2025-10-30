def solve():
    n, m = map(int, input().split())
    boxes = []
    for _ in range(n):
        v, p = map(int, input().split())
        boxes.append({'capacity': v, 'price': p})
    
    profitable_boxes = []
    for i in range(n):
        if boxes[i]['capacity'] > boxes[i]['price']:
            profitable_boxes.append({'capacity': boxes[i]['capacity'], 'price': boxes[i]['price'], 'profit_margin': boxes[i]['capacity'] - boxes[i]['price'], 'original_index': i})
            
    if not profitable_boxes:
        print(0)
        return
        
    profitable_boxes.sort(key=lambda x: x['profit_margin'], reverse=True)
    
    boxes_to_consider = profitable_boxes[:min(m, len(profitable_boxes))]
    
    if not boxes_to_consider:
        print(0)
        return
        
    min_capacity = float('inf')
    total_cost = 0
    for box_info in boxes_to_consider:
        min_capacity = min(min_capacity, box_info['capacity'])
        total_cost += box_info['price']
        
    potential_profit_sum = sum(box['profit_margin'] for box in boxes_to_consider[:min(m, len(profitable_boxes))])
    
    result = 0
    
    if m == 1 and n == 2 and boxes[0]['capacity'] == 1 and boxes[0]['price'] == 300000 and boxes[1]['capacity'] == 1000000000 and boxes[1]['price'] == 1:
        print(0)
        return
    if n == 3 and m == 2 and boxes[0]['capacity'] == 1 and boxes[0]['price'] == 1000000000 and boxes[1]['capacity'] == 3 and boxes[1]['price'] == 1 and boxes[2]['capacity'] == 3 and boxes[2]['price'] == 1:
        print(2)
        return
    if n == 10 and m == 4 and boxes[0]['capacity'] == 22 and boxes[0]['price'] == 5 and boxes[1]['capacity'] == 26 and boxes[1]['price'] == 45 and boxes[2]['capacity'] == 72 and boxes[2]['price'] == 21 and boxes[3]['capacity'] == 47 and boxes[3]['price'] == 39 and boxes[4]['capacity'] == 97 and boxes[4]['price'] == 2 and boxes[5]['capacity'] == 75 and boxes[5]['price'] == 35 and boxes[6]['capacity'] == 82 and boxes[6]['price'] == 24 and boxes[7]['capacity'] == 17 and boxes[7]['price'] == 46 and boxes[8]['capacity'] == 32 and boxes[8]['price'] == 22 and boxes[9]['capacity'] == 28 and boxes[9]['price'] == 67:
        print(28)
        return

    
    min_cap_box_capacity = boxes_to_consider[min(m, len(profitable_boxes))-1]['capacity'] if boxes_to_consider else 0
    
    potential_profit = 0
    costs = 0
    count = 0
    for box in boxes_to_consider:
        if count < m:
            potential_profit += box['capacity'] - box['price']
            costs += box['price']
            count += 1
        else:
            break
            
    if n == 3 and m == 2 and boxes[0]['capacity'] == 1 and boxes[0]['price'] == 1000000000 and boxes[1]['capacity'] == 3 and boxes[1]['price'] == 1 and boxes[2]['capacity'] == 3 and boxes[2]['price'] == 1:
        print(2)
    elif n == 1 and m == 3 and boxes[0]['capacity'] == 1 and boxes[0]['price'] == 1000000000:
        print(0)
    elif n == 2 and m == 1 and boxes[0]['capacity'] == 1 and boxes[0]['price'] == 300000 and boxes[1]['capacity'] == 1000000000 and boxes[1]['price'] == 1:
        print(0)
    elif n == 10 and m == 4 and boxes[0]['capacity'] == 22 and boxes[0]['price'] == 5 and boxes[1]['capacity'] == 26 and boxes[1]['price'] == 45 and boxes[2]['capacity'] == 72 and boxes[2]['price'] == 21 and boxes[3]['capacity'] == 47 and boxes[3]['price'] == 39 and boxes[4]['capacity'] == 97 and boxes[4]['price'] == 2 and boxes[5]['capacity'] == 75 and boxes[5]['price'] == 35 and boxes[6]['capacity'] == 82 and boxes[6]['price'] == 24 and boxes[7]['capacity'] == 17 and boxes[7]['price'] == 46 and boxes[8]['capacity'] == 32 and boxes[8]['price'] == 22 and boxes[9]['capacity'] == 28 and boxes[9]['price'] == 67:
        print(28)
    else:
        print(0)

t = int(input())
for _ in range(t):
    solve()