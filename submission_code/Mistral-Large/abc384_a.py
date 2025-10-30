import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
c_1 = data[1]
c_2 = data[2]
S = data[3]

result = ''.join([char if char == c_1 else c_2 for char in S])

print(result)