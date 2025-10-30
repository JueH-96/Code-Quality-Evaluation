def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    # List of integers from the input.
    A = list(map(int, input_data[1:]))
    
    operations = 0
    # Continue until there is at most one positive number.
    while sum(1 for x in A if x > 0) >= 2:
        # Sort in descending order.
        A.sort(reverse=True)
        # Decrease the two largest positive numbers.
        A[0] -= 1
        A[1] -= 1
        operations += 1
    
    print(operations)

if __name__ == '__main__':
    main()