import sys

def solve():
    # Initialize the stack with 100 cards, each labeled 0.
    # We'll use a Python list as a stack, where the end of the list is the top of the stack.
    # This means `append` adds to the top, and `pop` removes from the top.
    stack = [0] * 100

    # Read the number of queries
    Q = int(sys.stdin.readline())

    # Process each query
    for _ in range(Q):
        line = sys.stdin.readline().split()
        query_type = int(line[0])

        if query_type == 1:
            # Type 1: Place a card labeled with an integer x on top of the stack.
            x = int(line[1])
            stack.append(x)
        elif query_type == 2:
            # Type 2: Remove the top card of the stack and output the integer.
            # The problem guarantees the stack always has at least one card.
            removed_card_value = stack.pop()
            print(removed_card_value)

# Call the solve function to run the program
if __name__ == '__main__':
    solve()