# YOUR CODE HERE
import sys

def main():
    N, X, Y, Z = map(int, sys.stdin.read().split())
    
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