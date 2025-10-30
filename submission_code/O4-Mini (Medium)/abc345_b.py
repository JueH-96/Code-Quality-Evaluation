def main():
    import sys
    X = int(sys.stdin.readline().strip())
    # We want ceil(X/10). For any integer X,
    # ceil(X/10) == floor((X + 9) / 10) under integer arithmetic.
    result = (X + 9) // 10
    print(result)

if __name__ == "__main__":
    main()