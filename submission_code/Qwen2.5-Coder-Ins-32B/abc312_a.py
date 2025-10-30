# YOUR CODE HERE
import sys

S = sys.stdin.read().strip()

valid_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}

if S in valid_strings:
    print("Yes")
else:
    print("No")