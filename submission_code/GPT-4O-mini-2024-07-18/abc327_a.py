# YOUR CODE HERE
def has_adjacent_a_b(n, s):
    for i in range(n - 1):
        if (s[i] == 'a' and s[i + 1] == 'b') or (s[i] == 'b' and s[i + 1] == 'a'):
            return "Yes"
    return "No"

import sys
input = sys.stdin.read
data = input().splitlines()
n = int(data[0])
s = data[1]

print(has_adjacent_a_b(n, s))