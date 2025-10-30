# YOUR CODE HERE
import sys

def main():
    S = sys.stdin.read().strip()
    if S[0].isupper() and all(c.islower() for c in S[1:]):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()