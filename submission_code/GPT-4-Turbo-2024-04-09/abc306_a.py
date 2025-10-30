import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

result = ''.join([c * 2 for c in S])
print(result)