# YOUR CODE HERE
import sys
import re

N = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()

while True:
    new_S = re.sub(r'\([a-z]*\)', '', S)
    if new_S == S:
        break
    S = new_S

print(S)