# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
S = input[1]

result = ''.join([char * 2 for char in S])
print(result)