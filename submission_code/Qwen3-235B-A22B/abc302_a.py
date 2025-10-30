def main():
    import sys
    a, b = map(int, sys.stdin.read().split())
    print((a + b - 1) // b)

if __name__ == "__main__":
    main()