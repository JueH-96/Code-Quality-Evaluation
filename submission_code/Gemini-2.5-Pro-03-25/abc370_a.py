# YOUR CODE HERE
import sys

# Read input from stdin
# input().split() reads the line and splits by whitespace, returns list of strings
# map(int, ...) applies int() to each string in the list
# l, r = ... unpacks the resulting map object (or list) into variables l and r
l, r = map(int, sys.stdin.readline().split())

# Apply the logic based on the problem description

# Case 1: Snuke raises only his left hand (L=1, R=0).
# According to the instructions, this means he wants to eat takoyaki.
if l == 1 and r == 0:
    print("Yes")

# Case 2: Snuke raises only his right hand (L=0, R=1).
# According to the instructions, this means he does not want to eat takoyaki.
elif l == 0 and r == 1:
    print("No")

# Case 3: Snuke raises both hands (L=1, R=1) or no hands (L=0, R=0).
# In these cases, Snuke is not following the instructions for indicating preference.
# The problem states to print "Invalid" in these situations.
else: 
    print("Invalid")