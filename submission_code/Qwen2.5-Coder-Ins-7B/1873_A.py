# YOUR CODE HERE
import sys

input = sys.stdin.read
data = input().split()

t = int(data[0])
cases = data[1:]

for case in cases:
    if case == "abc":
        print("YES")
    elif case == "acb":
        print("YES")
    elif case == "bac":
        print("YES")
    elif case == "bca":
        print("NO")
    elif case == "cab":
        print("NO")
    elif case == "cba":
        print("YES")