def process_queries():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    Q = int(data[0])  # Number of queries
    stack = [0] * 100  # Initialize stack with 100 cards labeled 0
    top = 0  # Pointer to the top of the stack
    output = []
    
    for i in range(1, Q + 1):
        query = list(map(int, data[i].split()))
        if query[0] == 1:
            # Type 1 query: Place a card labeled with integer x on top of the stack
            x = query[1]
            stack[top] = x  # Place the card on top
            top += 1  # Move the top pointer up
        elif query[0] == 2:
            # Type 2 query: Remove the top card and output the integer on it
            top -= 1  # Move the top pointer down
            output.append(stack[top])  # Output the integer on the removed card
    
    # Print all outputs for type 2 queries
    print("
".join(map(str, output)))