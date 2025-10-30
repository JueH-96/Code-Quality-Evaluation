def main():
    import sys
    N, X, Y, Z = map(int, sys.stdin.readline().split())
    if (X < Y and X < Z < Y) or (X > Y and Y < Z < X):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()