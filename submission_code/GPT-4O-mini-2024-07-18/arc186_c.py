def calculate_money_increase(test_cases):
    results = []
    
    for case in test_cases:
        N, M, boxes = case
        # Sort boxes by their capacity and price
        boxes.sort(key=lambda x: (x[1], -x[0]))  # Sort by price, then by capacity descending
        
        total_money_increase = 0
        current_capacity = 0
        current_cost = 0
        
        for V, P in boxes:
            if current_capacity < V:
                # If we can still add balls to this box
                current_capacity = V
                current_cost += P
                total_money_increase += 1  # Each box can give us 1 yen for the first ball
            
            # If we reach the capacity, we can't add more of the same type
            if current_capacity == V:
                break
        
        total_money_increase -= current_cost  # Subtract the cost of boxes bought
        results.append(total_money_increase)
    
    return results

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    T = int(data[0])
    index = 1
    test_cases = []
    
    for _ in range(T):
        N, M = map(int, data[index].split())
        index += 1
        boxes = []
        for _ in range(N):
            V, P = map(int, data[index].split())
            boxes.append((V, P))
            index += 1
        test_cases.append((N, M, boxes))
    
    results = calculate_money_increase(test_cases)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()