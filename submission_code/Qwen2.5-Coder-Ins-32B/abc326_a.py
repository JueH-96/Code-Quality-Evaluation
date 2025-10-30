# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read().strip().split()
    X = int(input[0])
    Y = int(input[1])
    
    if abs(X - Y) <= 2 or abs(X - Y) <= 3:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()