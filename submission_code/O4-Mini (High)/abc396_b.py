def main():
    import sys
    input = sys.stdin.readline

    Q = int(input().strip())
    # Initialize the stack with 100 cards labeled 0
    stack = [0] * 100
    outputs = []

    for _ in range(Q):
        query = input().split()
        if query[0] == '1':
            # Type-1 query: push x on top
            x = int(query[1])
            stack.append(x)
        else:
            # Type-2 query: pop and record the top value
            val = stack.pop()
            outputs.append(val)

    # Output all popped values, one per line
    sys.stdout.write('
'.join(map(str, outputs)))

# Call main to execute
if __name__ == "__main__":
    main()