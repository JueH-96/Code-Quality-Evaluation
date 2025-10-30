# YOUR CODE HERE
import sys

# Read the input string from stdin
# .strip() removes any leading/trailing whitespace, including the newline character
S = sys.stdin.readline().strip()

# Sort the characters of the input string alphabetically.
# sorted(S) returns a list of characters, e.g., sorted("BAC") -> ['A', 'B', 'C']
sorted_S_list = sorted(S)

# Join the sorted list of characters back into a string.
# "".join(['A', 'B', 'C']) -> "ABC"
sorted_S_str = "".join(sorted_S_list)

# The target string, when its characters are sorted, is "ABC".
target_str = "ABC"

# Check if the sorted version of the input string S matches the target string "ABC".
# If they match, it means S contains exactly the characters 'A', 'B', 'C',
# and thus can be rearranged to form "ABC".
if sorted_S_str == target_str:
    # If it's possible to rearrange S to match "ABC", print "Yes"
    print("Yes")
else:
    # Otherwise, print "No"
    print("No")