def main():
    import sys
    
    # Read the two integers A and B
    data = sys.stdin.read().strip().split()
    if not data:
        return
    A, B = map(int, data)
    
    # Compute A^B + B^A
    result = pow(A, B) + pow(B, A)
    
    # Output the result
    print(result)


if __name__ == "__main__":
    main()