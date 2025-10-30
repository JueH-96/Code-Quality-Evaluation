import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    W = int(data[index])
    H = int(data[index + 1])
    index += 2

    N = int(data[index])
    index += 1

    strawberries = []
    for _ in range(N):
        p = int(data[index])
        q = int(data[index + 1])
        index += 2
        strawberries.append((p, q))

    A = int(data[index])
    index += 1

    a_cuts = [int(data[index + i]) for i in range(A)]
    index += A

    B = int(data[index])
    index += 1

    b_cuts = [int(data[index + i]) for i in range(B)]

    a_cuts = [0] + sorted(a_cuts) + [W]
    b_cuts = [0] + sorted(b_cuts) + [H]

    min_strawberries = float('inf')
    max_strawberries = 0

    for i in range(1, len(a_cuts)):
        for j in range(1, len(b_cuts)):
            x1, x2 = a_cuts[i-1], a_cuts[i]
            y1, y2 = b_cuts[j-1], b_cuts[j]

            count = sum(1 for p, q in strawberries if x1 < p < x2 and y1 < q < y2)

            min_strawberries = min(min_strawberries, count)
            max_strawberries = max(max_strawberries, count)

    print(min_strawberries, max_strawberries)

if __name__ == "__main__":
    solve()