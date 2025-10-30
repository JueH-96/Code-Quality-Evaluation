def solve():
    n = int(input())
    heights = list(map(int, input().split()))
    results = []
    for i in range(n):
        count = 0
        for j in range(i + 1, n):
            max_height_between = 0
            if j > i + 1:
                max_height_between = 0
                for k in range(i + 1, j):
                    max_height_between = max(max_height_between, heights[k])
            if max_height_between <= heights[j]:
                condition_met = True
            else:
                condition_met = False
            if condition_met:
                count += 1
        results.append(count)
    print(*(results))

if __name__ == '__main__':
    solve()