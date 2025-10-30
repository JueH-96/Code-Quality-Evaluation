import sys

def solve():
    N = int(sys.stdin.readline().strip())
    beans = []
    for i in range(N):
        A, C = map(int, sys.stdin.readline().strip().split())
        beans.append((A, C))

    beans.sort(key=lambda x: (x[1], -x[0]))

    max_deliciousness = 0
    min_deliciousness = float('inf')
    prev_color = -1
    for A, C in beans:
        if C != prev_color:
            min_deliciousness = A
        max_deliciousness = max(max_deliciousness, min_deliciousness)
        prev_color = C

    print(max_deliciousness)

solve()