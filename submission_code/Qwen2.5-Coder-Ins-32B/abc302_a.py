# YOUR CODE HERE
import sys

def main():
    A, B = map(int, sys.stdin.read().split())
    # Calculate the minimum number of attacks needed
    attacks = (A + B - 1) // B
    print(attacks)

if __name__ == "__main__":
    main()