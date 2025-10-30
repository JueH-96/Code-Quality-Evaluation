# YOUR CODE HERE
import sys

def solve():
    """
    Reads a 3-character string (a permutation of 'a', 'b', 'c')
    and determines if it can be transformed into "abc" using at most one swap.
    Outputs "YES" or "NO".
    """
    # Read the input string for the current test case
    s = sys.stdin.readline().strip()
    
    # The target string is "abc".
    # We are allowed AT MOST one swap (meaning 0 swaps or 1 swap).
    
    # Case 0 swaps: The string is already "abc".
    # If s == "abc", we need 0 swaps, which is <= 1. So, YES.
    
    # Case 1 swap: The string can become "abc" by swapping exactly two characters.
    # Let the input string be s[0]s[1]s[2]. The target is "abc".
    # A single swap exchanges the characters at two positions, say i and j.
    # The character at the third position k (where k is the index not equal to i or j)
    # remains unchanged.
    # For the string to become "abc" after one swap, the character at the
    # unchanged position k must already be the correct character for that
    # position in "abc".
    
    # This means at least one character in the original string must be
    # in its correct target position ('a' at index 0, 'b' at index 1, or 'c' at index 2).
    
    # If s[0] is 'a', the string is "a??" where ?? is a permutation of 'b' and 'c'.
    # It can be "abc" (0 swaps) or "acb" (1 swap). Both are possible.
    # If s[1] is 'b', the string is "?b?" where ?? is a permutation of 'a' and 'c'.
    # It can be "abc" (0 swaps) or "cba" (1 swap). Both are possible.
    # If s[2] is 'c', the string is "??c" where ?? is a permutation of 'a' and 'b'.
    # It can be "abc" (0 swaps) or "bac" (1 swap). Both are possible.
    
    # If NONE of s[0]=='a', s[1]=='b', s[2]=='c' are true, then all three characters
    # are in incorrect positions. Any single swap will move two characters, but the
    # third character, which was already in an incorrect position, will remain there.
    # Thus, the string cannot become "abc" with just one swap if all three characters
    # are initially misplaced.
    
    # Therefore, it is possible to make the string "abc" with at most one swap
    # if and only if at least one character is already in its correct target position.
    
    # Check if the character at index 0 is 'a' OR
    # the character at index 1 is 'b' OR
    # the character at index 2 is 'c'.
    
    # An alternative equivalent approach is to list all strings reachable
    # within at most one swap from "abc". These are the strings that
    # require 0 swaps ("abc") or 1 swap ("acb", "bac", "cba") to become "abc".
    # We can check if the input string is one of these.
    
    valid_start_strings = ["abc", "acb", "bac", "cba"]
    
    if s in valid_start_strings:
        print("YES")
    else:
        print("NO")

# Read the number of test cases
t = int(sys.stdin.readline())

# Process each test case
for _ in range(t):
    solve()

# END YOUR CODE HERE