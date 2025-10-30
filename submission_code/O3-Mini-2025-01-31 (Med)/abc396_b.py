def main():
    import sys
    input = sys.stdin.readline
    Q = int(input().strip())
    
    # initialize the stack with 100 cards labeled 0
    stack = [0] * 100
    
    results = []
    for _ in range(Q):
        parts = input().split()
        if parts[0] == '1':  # type 1 query: push the card x onto the stack
            x = int(parts[1])
            stack.append(x)
        else:  # type 2 query: pop the top card and output its value
            # Since the problem guarantees the stack is non-empty,
            # we can pop without checking:
            removed = stack.pop()
            results.append(removed)
    
    # Output the results, one per line
    sys.stdout.write("
".join(map(str, results)))
    
if __name__ == '__main__':
    main()