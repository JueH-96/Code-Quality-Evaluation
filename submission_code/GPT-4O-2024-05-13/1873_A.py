# YOUR CODE HERE
def can_be_abc(s):
    if s == "abc":
        return "YES"
    elif s == "acb" or s == "bac" or s == "cba":
        return "YES"
    else:
        return "NO"

import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
results = []

for i in range(1, t + 1):
    results.append(can_be_abc(data[i]))

for result in results:
    print(result)