def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, X, Y, Z = map(int, data)

    # If traveling in increasing order
    if X < Y:
        # Check if Z lies strictly between X and Y
        if X < Z < Y:
            print("Yes")
        else:
            print("No")
    else:
        # Traveling in decreasing order
        # Check if Z lies strictly between X and Y in decreasing sequence
        if Y < Z < X:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()