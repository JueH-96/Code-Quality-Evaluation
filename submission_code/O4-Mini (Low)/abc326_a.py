def main():
    import sys
    data = sys.stdin.read().strip().split()
    x, y = map(int, data)
    diff = y - x
    # If moving up two floors or less
    if diff > 0 and diff <= 2:
        print("Yes")
    # If moving down three floors or less
    elif diff < 0 and abs(diff) <= 3:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()