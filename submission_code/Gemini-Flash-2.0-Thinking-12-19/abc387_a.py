import sys

def solve():
    line = sys.stdin.readline().split()
    a = int(line[0])
    b = int(line[1])
    result = (a + b) ** 2
    print(result)

solve()