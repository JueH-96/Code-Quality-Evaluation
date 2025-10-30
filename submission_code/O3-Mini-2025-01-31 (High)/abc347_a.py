def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    # Parse numbers from input
    N = int(input_data[0])
    K = int(input_data[1])
    A = list(map(int, input_data[2:2+N]))
    
    # Filter elements that are multiples of K and divide them
    result = [a // K for a in A if a % K == 0]
    # Sort the result in ascending order (although A is given in ascending order, this ensures correctness)
    result.sort()
    
    # Print the results with spaces in between
    print(" ".join(map(str, result)))

if __name__ == '__main__':
    main()