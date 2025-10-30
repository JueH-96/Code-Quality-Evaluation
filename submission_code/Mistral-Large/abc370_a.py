import sys
input = sys.stdin.read
data = input().split()

L = int(data[0])
R = int(data[1])

if L == 1 and R == 0:
    print("Yes")
elif L == 0 and R == 1:
    print("No")
else:
    print("Invalid")