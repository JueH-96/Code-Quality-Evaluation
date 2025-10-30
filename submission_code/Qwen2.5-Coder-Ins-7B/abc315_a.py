# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

S = data[0]

result = ''.join([char for char in S if char not in 'aeiou'])

print(result)