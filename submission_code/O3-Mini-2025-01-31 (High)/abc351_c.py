def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))
    
    # We'll use a stack to simulate adding the balls and merging them.
    # Each ball is represented by its exponent.
    stack = []
    for a in A:
        stack.append(a)
        # While there are at least two balls and the two rightmost have the same exponent, merge them.
        while len(stack) >= 2 and stack[-1] == stack[-2]:
            # Pop the two equal balls.
            x = stack.pop()
            stack.pop()
            # Their sum is equivalent to a ball of exponent (x + 1) because 
            # 2^x + 2^x = 2 * 2^x = 2^(x+1).
            stack.append(x + 1)
    
    # Print the final number of balls
    sys.stdout.write(str(len(stack)))

if __name__ == '__main__':
    main()