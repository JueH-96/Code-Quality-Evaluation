# YOUR CODE HERE
def solve_case():
    N, M = map(int, input().split())
    boxes = [tuple(map(int, input().split())) for _ in range(N)]
    
    # Sort boxes by price in ascending order
    boxes.sort(key=lambda x: x[1])
    
    max_profit = 0
    current_capacity = 0
    current_cost = 0
    
    for capacity, price in boxes:
        # Calculate potential profit for this box
        potential_profit = min(capacity, M) - price
        
        # If this box can increase profit, use it
        if potential_profit > 0:
            max_profit += potential_profit
            current_capacity += capacity
            current_cost += price
            
            # If we can accommodate all ball types, we're done
            if current_capacity >= M:
                break
    
    return max_profit

T = int(input())
for _ in range(T):
    print(solve_case())