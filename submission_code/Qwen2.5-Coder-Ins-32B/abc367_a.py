# YOUR CODE HERE
import sys

def main():
    A, B, C = map(int, sys.stdin.read().split())
    
    if B < C:
        if A >= B and A < C:
            print("Yes")
        else:
            print("No")
    else:
        if A >= B or A < C:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()