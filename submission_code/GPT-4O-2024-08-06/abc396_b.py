# YOUR CODE HERE
def process_queries(queries):
    # Initialize the stack with 100 cards labeled 0
    stack = [0] * 100
    output = []
    
    for query in queries:
        if query[0] == 1:
            # Type 1 query: Place a card labeled with an integer x on top of the stack
            x = query[1]
            stack.append(x)
        elif query[0] == 2:
            # Type 2 query: Remove the top card of the stack and output the integer
            top_card = stack.pop()
            output.append(top_card)
    
    return output

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().strip().split('
')
    
    Q = int(data[0])
    queries = []
    
    for i in range(1, Q + 1):
        parts = list(map(int, data[i].split()))
        queries.append(parts)
    
    results = process_queries(queries)
    
    for result in results:
        print(result)