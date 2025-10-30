def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # The first token is N
    n = int(input_data[0])
    A = []
    index = 1
    # Build the triangular matrix A.
    # The i-th row (1-indexed) contains i integers.
    for i in range(1, n + 1):
        row = list(map(int, input_data[index:index + i]))
        A.append(row)
        index += i

    # Starting element is 1.
    current = 1
    # Combine with each element from 1 to N in order.
    for element in range(1, n + 1):
        if current >= element:
            current = A[current - 1][element - 1]
        else:
            current = A[element - 1][current - 1]
    
    sys.stdout.write(str(current))

if __name__ == '__main__':
    main()