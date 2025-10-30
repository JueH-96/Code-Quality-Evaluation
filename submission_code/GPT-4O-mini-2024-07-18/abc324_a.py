# YOUR CODE HERE
def all_equal(numbers):
    return all(x == numbers[0] for x in numbers)

import sys

input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
A = list(map(int, data[1].split()))

if all_equal(A):
    print("Yes")
else:
    print("No")