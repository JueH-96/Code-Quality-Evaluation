def process():
    rack = [[0] * 10 for _ in range(10)]
    column = [[0] * 10 for _ in range(10)]
    groups = [[0] * 10 for _ in range(10)]
    for i in range(1, 10):
        line = list(map(int, input().split()))
        for j in range(9):
            num = line[j]
            rack[i][num] += 1
            column[j][num] += 1
            group = (i - 1) // 3 * 3 + (j // 3) + 1
            groups[group][num] += 1
            if rack[i][num] > 1 or column[j][num] > 1 or groups[group][num] > 1:
                print('No')
                return
    print('Yes')

process()