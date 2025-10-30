import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

position = S.find("ABC")
if position == -1:
    print(-1)
else:
    print(position + 1)