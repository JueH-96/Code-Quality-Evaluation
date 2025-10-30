def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, X, Y, Z = map(int, data)

    # If traveling inbound (X < Y), we go from X up to Y.
    # If traveling outbound (X > Y), we go from X down to Y.
    # In either case, check if Z lies between X and Y inclusive.
    if (X < Y and X <= Z <= Y) or (X > Y and Y <= Z <= X):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()