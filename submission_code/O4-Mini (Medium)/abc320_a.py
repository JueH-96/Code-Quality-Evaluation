def main():
    import sys
    data = sys.stdin.read().strip().split()
    A, B = map(int, data)
    # Compute A^B + B^A
    result = pow(A, B) + pow(B, A)
    print(result)

if __name__ == "__main__":
    main()