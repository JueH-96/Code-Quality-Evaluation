def main():
    import sys
    X = int(sys.stdin.readline().strip())
    # For any integer X, the ceiling of X/10 can be computed as (X + 9) // 10,
    # even when X is negative, because Python's // operator floors toward -infinity.
    result = (X + 9) // 10
    print(result)

if __name__ == "__main__":
    main()