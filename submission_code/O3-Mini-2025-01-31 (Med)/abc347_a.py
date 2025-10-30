def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    # Read first two integers
    N = int(input_data[0])
    K = int(input_data[1])
    # Read the sequence A
    A = list(map(int, input_data[2:2+N]))
    
    # Filter elements that are multiples of K, compute quotient
    quotients = [a // K for a in A if a % K == 0]
    
    # Sort the quotients in ascending order (though A is already sorted,
    # performing sort to be sure)
    quotients.sort()
    
    # Print the result in one line separated by spaces
    print(" ".join(map(str, quotients)))

if __name__ == '__main__':
    main()