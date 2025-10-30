# YOUR CODE HERE
import sys

def can_be_abc(s):
    if s == "abc":
        return True
    # Check swap positions 0 and 1
    if s[1] + s[0] + s[2] == "abc":
        return True
    # Check swap positions 0 and 2
    if s[2] + s[1] + s[0] == "abc":
        return True
    # Check swap positions 1 and 2
    if s[0] + s[2] + s[1] == "abc":
        return True
    return False

data = sys.stdin.read().split()
index = 0
t = int(data[index])
index += 1
for _ in range(t):
    s = data[index]
    index += 1
    if can_be_abc(s):
        print("YES")
    else:
        print("NO")