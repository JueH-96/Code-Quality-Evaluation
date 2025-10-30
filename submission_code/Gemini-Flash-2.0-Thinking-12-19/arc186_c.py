def solve():
    n, m = map(int, input().split())
    boxes = []
    for _ in range(n):
        v_i, p_i = map(int, input().split())
        boxes.append({'capacity': v_i, 'price': p_i})
    
    profits = []
    for box in boxes:
        profit = box['capacity'] - box['price']
        if profit > 0:
            profits.append(profit)
            
    if not profits:
        print(0)
        return
        
    if m == 1:
        print(sum(profits))
    else:
        profits.sort(reverse=True)
        total_profit = 0
        denominator = 2**(m-1)
        for i in range(min(len(profits), m)):
            total_profit += profits[i] // denominator
        print(total_profit)

t = int(input())
for _ in range(t):
    solve()