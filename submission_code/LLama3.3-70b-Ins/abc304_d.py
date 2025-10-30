import sys

def count_strawberries(x, y, strawberries, a, b):
    count = 0
    for strawberry in strawberries:
        if a[x] < strawberry[0] <= a[x+1] and b[y] < strawberry[1] <= b[y+1]:
            count += 1
    return count

def main():
    W, H = map(int, sys.stdin.readline().split())
    N = int(sys.stdin.readline())
    strawberries = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    A = int(sys.stdin.readline())
    a = [0] + list(map(int, sys.stdin.readline().split())) + [W]
    B = int(sys.stdin.readline())
    b = [0] + list(map(int, sys.stdin.readline().split())) + [H]

    min_strawberries = float('inf')
    max_strawberries = 0

    for x in range(A+1):
        for y in range(B+1):
            count = count_strawberries(x, y, strawberries, a, b)
            min_strawberries = min(min_strawberries, count)
            max_strawberries = max(max_strawberries, count)

    print(min_strawberries, max_strawberries)

if __name__ == "__main__":
    main()