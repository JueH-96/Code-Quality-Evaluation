def main():
    import sys
    input_data = sys.stdin.read().split()
    Q = int(input_data[0])
    # Initialize the stack with 100 cards of value 0
    stack = [0] * 100
    result = []
    index = 1
    for _ in range(Q):
        query_type = int(input_data[index])
        index += 1
        if query_type == 1:
            # Place x on top of the stack
            x = int(input_data[index])
            index += 1
            stack.append(x)
        else:  # query_type == 2
            # Remove the top card and record its value
            value = stack.pop()
            result.append(value)
    # Output the results for each "remove" operation
    sys.stdout.write("
".join(map(str, result)) + "
")
    
if __name__ == '__main__':
    main()