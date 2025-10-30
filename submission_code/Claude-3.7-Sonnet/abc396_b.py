def solve():
    # Initialize a stack with 100 cards labeled 0
    stack = [0] * 100
    
    # Read the number of queries
    Q = int(input())
    
    # Process each query
    for _ in range(Q):
        query = input().split()
        
        if query[0] == '1':
            # Type 1: Add a card with value x to the top
            x = int(query[1])
            stack.append(x)
        elif query[0] == '2':
            # Type 2: Remove and output the top card
            top_card = stack.pop()
            print(top_card)

solve()