def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    n = int(input_data[0])
    numbers = list(map(int, input_data[1:]))
    
    # We'll use a list as a stack. We'll store the exponent representing the ball size 2^exp.
    stack = []
    
    for a in numbers:
        # Add the new ball (its exponent is given by a)
        stack.append(a)
        # While there are at least two balls and the two rightmost ones are same
        while len(stack) >= 2 and stack[-1] == stack[-2]:
            # Remove the two balls
            x = stack.pop()
            stack.pop()
            # Create a new ball with exponent x+1 (since 2^x + 2^x = 2^(x+1))
            stack.append(x+1)
    
    # The answer is simply the number of balls remaining.
    print(len(stack))

if __name__ == '__main__':
    main()