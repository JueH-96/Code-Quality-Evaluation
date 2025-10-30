# YOUR CODE HERE
import sys
input = sys.stdin.read

A, B = map(int, input().split())
sum_ab = A + B

# We need to print any integer between 0 and 9 that is not equal to sum_ab
# We can simply choose 0 if sum_ab is not 0, otherwise choose 1
if sum_ab != 0:
    print(0)
else:
    print(1)