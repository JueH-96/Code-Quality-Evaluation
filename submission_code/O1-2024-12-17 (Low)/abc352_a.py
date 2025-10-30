def main():
    import sys
    N, X, Y, Z = map(int, sys.stdin.readline().strip().split())
    
    # If X < Y, Takahashi travels on an inbound train (stations increase)
    if X < Y:
        # Check if Z is between X and Y
        print("Yes" if X < Z < Y else "No")
    else:
        # Otherwise, Takahashi travels on an outbound train (stations decrease)
        # Check if Z is between Y and X
        print("Yes" if Y < Z < X else "No")

# Don't forget to call main()
if __name__ == "__main__":
    main()