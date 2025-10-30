# YOUR CODE HERE
import sys

S = sys.stdin.readline().strip()

# Check if the contest number is between 001 and 349, excluding 316
contest_number = int(S[3:])
if 1 <= contest_number <= 349 and contest_number != 316:
    print("Yes")
else:
    print("No")