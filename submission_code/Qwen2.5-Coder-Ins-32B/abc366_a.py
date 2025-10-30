# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    T = int(input[1])
    A = int(input[2])
    
    if T > (N - T - A) + T:
        print("Yes")
    elif A > (N - T - A) + A:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()