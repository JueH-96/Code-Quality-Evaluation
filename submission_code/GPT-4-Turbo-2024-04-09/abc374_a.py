# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    S = input().strip()
    if S.endswith("san"):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()