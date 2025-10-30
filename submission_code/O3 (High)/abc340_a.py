def main():
    import sys
    
    # Read input
    data = sys.stdin.read().strip().split()
    if not data:
        return
    A, B, D = map(int, data)
    
    # Generate the arithmetic sequence
    sequence = range(A, B + 1, D)
    
    # Print results separated by spaces
    print(" ".join(map(str, sequence)))

if __name__ == "__main__":
    main()