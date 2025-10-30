def solve_test_case(N, M, boxes):
    # Sort boxes by price
    boxes.sort(key=lambda x: x[1])
    
    # For each box, calculate maximum profit possible
    max_profit = 0
    for i in range(N):
        capacity = boxes[i][0]
        cost = boxes[i][1]
        # We can earn capacity yen by filling the box
        # but need to pay cost to buy the box
        profit = capacity - cost
        if profit > max_profit:
            max_profit = profit
            
    return max(0, max_profit)

def main():
    # Read number of test cases
    T = int(input())
    
    # Process each test case
    for _ in range(T):
        # Read N and M
        N, M = map(int, input().split())
        
        # Read box information
        boxes = []
        for _ in range(N):
            V, P = map(int, input().split())
            boxes.append((V, P))
            
        # Solve and print result
        result = solve_test_case(N, M, boxes)
        print(result)

if __name__ == "__main__":
    main()