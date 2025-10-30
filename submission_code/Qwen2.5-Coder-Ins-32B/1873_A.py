# YOUR CODE HERE
import sys

def can_make_abc(s):
    if s == "abc":
        return "YES"
    if s == "acb":
        return "YES"
    if s == "bac":
        return "YES"
    if s == "bca":
        return "NO"
    if s == "cab":
        return "NO"
    if s == "cba":
        return "YES"
    return "NO"

input = sys.stdin.read().strip().split('
')
t = int(input[0])
results = []

for i in range(1, t + 1):
    s = input[i]
    results.append(can_make_abc(s))

for result in results:
    print(result)