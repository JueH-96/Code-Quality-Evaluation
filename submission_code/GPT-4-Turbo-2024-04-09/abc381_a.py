# YOUR CODE HERE
def is_1122_string(s):
    n = len(s)
    if n % 2 == 0:
        return False
    mid_index = (n // 2)
    if s[mid_index] != '/':
        return False
    if not all(c == '1' for c in s[:mid_index]):
        return False
    if not all(c == '2' for c in s[mid_index+1:]):
        return False
    return True

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

if is_1122_string(S):
    print("Yes")
else:
    print("No")