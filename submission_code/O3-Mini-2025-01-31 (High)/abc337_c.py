def main():
    import sys
    input = sys.stdin.readline
    
    # Read number of persons
    n = int(input().strip())
    # Read the arrangement list where A[i] indicates the person that person (i+1) is behind or -1 if front.
    A = list(map(int, input().split()))
    
    # Create a dictionary mapping a person to the person standing directly behind them.
    next_person = {}
    front = None
    
    # Process the input list.
    for i in range(1, n+1):
        if A[i-1] == -1:
            front = i  # Found the person at the front of the line.
        else:
            # Person i is right behind person A[i-1]
            next_person[A[i-1]] = i
    
    # Build and collect the chain starting from the front person.
    order = []
    current = front
    while True:
        order.append(current)
        if current in next_person:
            current = next_person[current]
        else:
            break
    
    # Output the arranged order.
    print(" ".join(map(str, order)))
    
if __name__ == '__main__':
    main()