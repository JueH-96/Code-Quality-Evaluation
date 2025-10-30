import sys

def solve():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    count = 0
    while True:
        A.sort(reverse=True)
        if A[1] == 0:
            break
        A[0] -= 1
        A[1] -= 1
        count += 1

    print(count)

solve()