# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    Z = int(data[3])
    
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