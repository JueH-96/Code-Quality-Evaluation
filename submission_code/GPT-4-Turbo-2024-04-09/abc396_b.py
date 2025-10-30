import sys
input = sys.stdin.read

def process_queries():
    data = input().split()
    Q = int(data[0])
    queries = data[1:]
    
    stack = [0] * 100  # Initial stack with 100 cards labeled 0
    top_index = 99  # Points to the top card of the stack
    
    output = []
    i = 0
    while i < len(queries):
        if queries[i] == '2':
            # Type 2 query: remove the top card and output its value
            output.append(str(stack[top_index]))
            top_index -= 1
        elif queries[i] == '1':
            # Type 1 query: add a card with value x on top
            x = int(queries[i + 1])
            top_index += 1
            stack[top_index] = x
            i += 1  # Skip the next item because it's the value x
        i += 1
    
    print("
".join(output))

process_queries()