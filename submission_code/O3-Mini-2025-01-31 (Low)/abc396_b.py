def main():
    import sys
    input = sys.stdin.readline

    Q = int(input().strip())
    # Create initial stack of 100 zeros.
    stack = [0] * 100

    outputs = []
    for _ in range(Q):
        query = input().split()
        if query[0] == '1':
            # Push card x
            x = int(query[1])
            stack.append(x)
        elif query[0] == '2':
            # Pop card from the stack and output its value.
            outputs.append(str(stack.pop()))
    
    # Print all outputs
    print("
".join(outputs))

if __name__ == '__main__':
    main()