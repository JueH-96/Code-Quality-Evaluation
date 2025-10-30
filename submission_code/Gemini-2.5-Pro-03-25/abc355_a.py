import sys

# Read input A and B from standard input
# input().split() reads a line from stdin, splits it by whitespace, returning a list of strings.
# map(int, ...) applies the int() function to each string element in the list, converting them to integers.
# a, b = ... unpacks the resulting map object (or list in Python 2) into variables a and b.
a, b = map(int, sys.stdin.readline().split())

# The set of all suspects initially is {1, 2, 3}.
# Witness 1 (Ringo) remembers that person A is not the culprit.
# Witness 2 (Snuke) remembers that person B is not the culprit.
# Based on these memories, the set of potential culprits is {1, 2, 3} excluding A and excluding B.
# Mathematically, the set of possible culprits is P = {1, 2, 3} \ {A, B}.

# We need to determine if the size of set P is exactly 1.

# Case 1: A and B are the same person (A == B).
if a == b:
    # If A and B are the same, only one person is ruled out.
    # The set of potential culprits is P = {1, 2, 3} \ {A}.
    # The size of P is 3 - 1 = 2.
    # Example: If A=B=1, potential culprits are {2, 3}.
    # Since there are two possibilities, the culprit cannot be uniquely identified.
    # In this case, we should output -1.
    print("-1")
# Case 2: A and B are different people (A != B).
else:
    # If A and B are different, two distinct people are ruled out.
    # The set of potential culprits is P = {1, 2, 3} \ {A, B}.
    # Since A and B are distinct elements from {1, 2, 3}, the size of {A, B} is 2.
    # The size of P is 3 - 2 = 1.
    # Example: If A=1, B=2, the potential culprit is {3}.
    # Since there is exactly one possibility, the culprit is uniquely identified.
    
    # We need to find the single element in P.
    # Method 1: Iterate through {1, 2, 3} and find the one not equal to A and B.
    # for i in range(1, 4):
    #     if i != a and i != b:
    #         print(i)
    #         break # Found the unique culprit

    # Method 2: Use set operations.
    # candidates = {1, 2, 3}
    # candidates.difference_update({a, b}) # Remove a and b from the set
    # print(list(candidates)[0]) # Convert set to list to get the element

    # Method 3: Use the sum property.
    # The sum of the three suspects' numbers is 1 + 2 + 3 = 6.
    # Let C be the number of the actual culprit.
    # Since A, B are not the culprits and A != B, the set of people {A, B, C} must be exactly {1, 2, 3}.
    # Therefore, the sum A + B + C must equal the sum 1 + 2 + 3.
    # A + B + C = 6
    # So, the culprit's number C can be calculated as C = 6 - A - B.
    culprit = 6 - a - b
    print(culprit)