def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    stack = []
    for a in A:
        stack.append(a)
        # Check and combine balls while the last two have the same exponent.
        while len(stack) >= 2 and stack[-1] == stack[-2]:
            v = stack.pop()
            stack.pop()
            stack.append(v + 1)
            
    sys.stdout.write(str(len(stack)))

if __name__ == '__main__':
    main()