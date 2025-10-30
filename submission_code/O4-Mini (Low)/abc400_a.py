def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    A = int(data)
    # We need B such that A * B = 400
    # That means B = 400 / A must be an integer.
    if 400 % A == 0:
        print(400 // A)
    else:
        print(-1)

if __name__ == "__main__":
    main()