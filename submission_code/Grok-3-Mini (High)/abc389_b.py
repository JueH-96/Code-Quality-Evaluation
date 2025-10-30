import sys
data = sys.stdin.read().strip()
X = int(data)
N = 2
fact = 2
while fact < X:
    N += 1
    fact *= N
print(N)