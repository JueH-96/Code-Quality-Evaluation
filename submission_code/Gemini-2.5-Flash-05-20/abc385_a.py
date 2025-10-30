import sys

def solve():
    # Read the three integers A, B, C from standard input.
    # map(int, ...) converts the string parts to integers.
    # sys.stdin.readline().split() reads a line and splits it by whitespace.
    A, B, C = map(int, sys.stdin.readline().split())

    # Check for possible divisions into two groups with equal sums.
    # If A is equal to the sum of B and C, then (A) and (B,C) form two groups with equal sums.
    # Same logic applies if B equals (A+C) or C equals (A+B).
    if (A == B + C) or \
       (B == A + C) or \
       (C == A + B):
        print("Yes")
        return # If found, no need to check further

    # Check for possible division into three groups with equal sums.
    # This is only possible if all three numbers are identical.
    # If A, B, and C are all equal, then (A), (B), (C) form three groups with equal sums.
    if (A == B and B == C):
        print("Yes")
        return # If found, no need to check further

    # If none of the above conditions are met, it's not possible.
    print("No")

# Call the solve function to execute the logic.
solve()