def main():
    X, Y = map(int, input().split())
    diff = Y - X
    # If going up at most 2 floors or going down at most 3 floors, use stairs
    if (diff > 0 and diff <= 2) or (diff < 0 and -diff <= 3):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()