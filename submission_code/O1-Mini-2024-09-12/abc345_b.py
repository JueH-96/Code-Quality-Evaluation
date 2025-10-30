import sys

X = int(sys.stdin.read())

if X >= 0:
    result = (X + 9) // 10
else:
    if X % 10 == 0:
        result = X // 10
    else:
        result = X // 10 + 1

print(result)