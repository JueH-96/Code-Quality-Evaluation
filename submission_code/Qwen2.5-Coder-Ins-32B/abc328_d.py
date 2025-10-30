# YOUR CODE HERE
import sys

def main():
    S = sys.stdin.read().strip()
    while "ABC" in S:
        S = S.replace("ABC", "", 1)
    print(S)

if __name__ == "__main__":
    main()