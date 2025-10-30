# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    N, X, Y, Z = map(int, data.split())
    
    # Determine the direction of travel
    if X < Y:
        # Inbound train
        if X < Z < Y:
            print("Yes")
        else:
            print("No")
    else:
        # Outbound train
        if Y < Z < X:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()