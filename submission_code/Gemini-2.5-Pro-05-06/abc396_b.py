def solve():
    # Initialize the stack with 100 cards, each labeled 0.
    # The end of the list will represent the top of the stack.
    stack = [0] * 100

    # Read the number of queries
    Q = int(input())

    # Process each query
    for _ in range(Q):
        query_parts = input().split()
        query_type = int(query_parts[0])

        if query_type == 1:
            # Type 1: Place a card with label x on top
            x = int(query_parts[1])
            stack.append(x)
        elif query_type == 2:
            # Type 2: Remove the top card and output its label
            # The problem guarantees the stack is not empty before this operation.
            removed_card_label = stack.pop()
            print(removed_card_label)

# Call the function to execute the solution logic
solve()