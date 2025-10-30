def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    X = int(data[0])
    Y = int(data[1])
    Z = int(data[2])
    S = data[3]
    
    N = len(S)
    
    # Initialize previous costs
    prev_0 = 0  # Caps Lock off
    prev_1 = Z  # Caps Lock on (cost to toggle once)
    
    for c in S:
        # Cost to type character c with Caps Lock off
        if c == 'a':
            cost_0 = X
        else:  # c == 'A'
            cost_0 = Y
        # Cost to type character c with Caps Lock on
        if c == 'A':
            cost_1 = X
        else:  # c == 'a'
            cost_1 = Y
        
        # Calculate current costs
        curr_0 = min(prev_0 + cost_0, prev_1 + Z + cost_0)
        curr_1 = min(prev_0 + Z + cost_1, prev_1 + cost_1)
        
        # Update previous costs for next iteration
        prev_0, prev_1 = curr_0, curr_1
    
    # The answer is the minimum of the two possible final states
    print(min(prev_0, prev_1))

if __name__ == '__main__':
    main()