def main():
    import sys
    input = sys.stdin.readline
    
    Q = int(input())
    # Initialize stack with 100 cards labeled 0
    stack = [0] * 100
    
    outputs = []
    for _ in range(Q):
        query = input().split()
        c = query[0]
        if c == '1':
            # Push x onto the stack
            x = int(query[1])
            stack.append(x)
        else:
            # Pop and record the top card
            outputs.append(str(stack.pop()))
    
    # Print all pop results, one per line
    sys.stdout.write("
".join(outputs))

if __name__ == "__main__":
    main()