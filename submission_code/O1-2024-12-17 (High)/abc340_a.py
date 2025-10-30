def main():
    A, B, D = map(int, input().split())
    # Generate and print the arithmetic sequence
    sequence = range(A, B+1, D)
    print(*sequence)

if __name__ == "__main__":
    main()