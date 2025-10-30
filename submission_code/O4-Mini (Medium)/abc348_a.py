def main():
    import sys
    
    # Read the number of penalty kicks
    N = int(sys.stdin.readline().strip())
    
    # Build the result string: 'x' if i is a multiple of 3, 'o' otherwise
    result = []
    for i in range(1, N + 1):
        if i % 3 == 0:
            result.append('x')
        else:
            result.append('o')
    
    # Output the joined string
    print(''.join(result))


if __name__ == "__main__":
    main()