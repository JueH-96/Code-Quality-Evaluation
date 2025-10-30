from collections import defaultdict

def main():
    H, W, M = map(int, input().split())
    row_colors = defaultdict(set)
    col_colors = defaultdict(set)
    color_count = defaultdict(int)
    color_set = set()

    for _ in range(M):
        T, A, X = map(int, input().split())
        if T == 1:
            row_colors[A].clear()
            row_colors[A].add(X)
            for col in range(1, W + 1):
                color_set.discard((A, col))
                color_set.add((A, col, X))
                color_count[X] += 1
                color_count[(A, col)] = 0
        elif T == 2:
            col_colors[A].clear()
            col_colors[A].add(X)
            for row in range(1, H + 1):
                color_set.discard((row, A))
                color_set.add((row, A, X))
                color_count[X] += 1
                color_count[(row, A)] = 0

    for row in range(1, H + 1):
        for col in range(1, W + 1):
            if (row, col) not in color_set:
                color_set.add((row, col))
                color = 0
                for x in row_colors[row]:
                    color = x
                for x in col_colors[col]:
                    color = x
                color_count[color] += 1

    K = len(color_count)
    print(K)
    for color, count in sorted(color_count.items()):
        if count > 0:
            print(color, count)

if __name__ == "__main__":
    main()