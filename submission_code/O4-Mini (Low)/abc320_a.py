def main():
    import sys
    data = sys.stdin.read().strip().split()
    A, B = map(int, data)
    # Compute A^B + B^A using the exponentiation operator **
    result = A**B + B**A
    print(result)

if __name__ == "__main__":
    main()