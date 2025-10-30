import sys
A, B = map(int, sys.stdin.readline().split())
answer = (A + B - 1) // B
print(answer)