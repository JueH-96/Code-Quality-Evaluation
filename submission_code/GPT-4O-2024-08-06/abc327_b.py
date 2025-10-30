# YOUR CODE HERE
import sys
import math

def find_A_for_B(B):
    # We need to find A such that A^A = B
    # A^A grows very fast, so we can limit A to a reasonable range
    # A^A <= B implies A <= B^(1/A)
    
    # We will try A from 1 to a reasonable upper bound
    # A^A grows very fast, so we don't need to check very large A
    # Let's consider A up to 60, because 60^60 is already a very large number
    for A in range(1, 61):
        if A ** A == B:
            return A
    return -1

def main():
    input = sys.stdin.read
    B = int(input().strip())
    result = find_A_for_B(B)
    print(result)

if __name__ == "__main__":
    main()