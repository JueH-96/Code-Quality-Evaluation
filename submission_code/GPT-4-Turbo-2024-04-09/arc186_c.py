def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        M = int(data[index + 1])
        index += 2
        
        boxes = []
        for _ in range(N):
            V = int(data[index])
            P = int(data[index + 1])
            index += 2
            boxes.append((V, P))
        
        # Sort boxes by price ascending
        boxes.sort(key=lambda x: x[1])
        
        # We will calculate the maximum possible profit
        max_profit = 0
        total_cost = 0
        
        # We can use each box to its full capacity for each type of ball
        # Since Mr. Ball can always choose a ball type that is not in a box to force the game to end
        # if Mr. Box tries to put it in a filled box of another type.
        # Thus, Mr. Box should only use a box for one type of ball to maximize profit.
        
        for capacity, price in boxes:
            if price > total_cost + capacity:
                # If the price of the next box is greater than what we can cover with the current capacity profit,
                # we stop buying more boxes.
                break
            total_cost += price
            max_profit += capacity
        
        # The profit is the total balls placed minus the cost spent on boxes
        results.append(max_profit - total_cost)
    
    for result in results:
        print(result)