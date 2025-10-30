# YOUR CODE HERE
import sys

S = sys.stdin.read().strip()

if S.startswith("ABC") and S[3:].isdigit():
    num = int(S[3:])
    if 1 <= num <= 314 or num == 315 or (317 <= num <= 348):
        print("Yes")
    else:
        print("No")
else:
    print("No")