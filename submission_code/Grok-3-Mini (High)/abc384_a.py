import sys
data = sys.stdin.read().split()
c1 = data[1]
c2 = data[2]
S = data[3]
result = ''.join([c2 if char != c1 else char for char in S])
print(result)