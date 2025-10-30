# YOUR CODE HERE
import sys

# Function to solve a single test case
def solve():
    """
    Reads a 3-character string permutation of 'abc', 
    determines if it can be transformed into 'abc' with at most one swap,
    and prints "YES" or "NO".
    """
    # Read the input string for the test case from standard input
    s = sys.stdin.readline().strip()
    
    # Define the target string
    target = "abc"
    
    # A string can be transformed into the target string "abc" with at most one swap if:
    # 1. The string is already "abc" (0 swaps needed). This means 0 positions differ.
    # 2. The string differs from "abc" by exactly one swap. A single swap affects exactly two positions. 
    #    Therefore, the string must differ from "abc" in exactly 2 positions.
    #
    # If the string differs from "abc" in 3 positions, it requires more than one swap.
    # It's impossible for two permutations of the same set of elements to differ in exactly 1 position.
    # So, we only need to check if the number of differing positions is 0 or 2.

    # Calculate the number of positions where the input string 's' differs from the target string 'abc'.
    # We can iterate through the indices 0, 1, 2 and count the mismatches.
    differences = 0
    if s[0] != target[0]:
        differences += 1
    if s[1] != target[1]:
        differences += 1
    if s[2] != target[2]:
        differences += 1
        
    # Alternatively, using a more concise generator expression and sum:
    # differences = sum(1 for i in range(3) if s[i] != target[i])
        
    # Check the condition based on the number of differences.
    if differences == 0 or differences == 2:
        # If 0 differences (s == target) or 2 differences (s can become target with one swap)
        print("YES")
    else: # differences == 3
        # If 3 differences, it's not possible with at most one swap.
        print("NO")

# Read the number of test cases from the first line of standard input.
t = int(sys.stdin.readline())

# Process each test case by calling the solve function.
for _ in range(t):
    solve()