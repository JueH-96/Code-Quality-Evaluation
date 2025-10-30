import sys

def solve():
    line = sys.stdin.readline().split()
    A = int(line[0])
    B = int(line[1])

    term1 = A ** B
    term2 = B ** A

    result = term1 + term2
    print(result)

if __name__ == '__main__':
    solve()