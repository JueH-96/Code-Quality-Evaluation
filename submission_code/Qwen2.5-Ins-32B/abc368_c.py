import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    h = list(map(int, data[1:]))
    
    t = 0
    for i in range(n):
        while h[i] > 0:
            t += 1
            if t % 3 == 0:
                h[i] -= 3
            else:
                h[i] -= 1
    print(t)

solve()