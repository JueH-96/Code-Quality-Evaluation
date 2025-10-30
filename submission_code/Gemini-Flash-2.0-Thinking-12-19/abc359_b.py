def solve():
    n = int(input())
    a = list(map(int, input().split()))
    count = 0
    for color in range(1, n + 1):
        indices = []
        for i in range(2 * n):
            if a[i] == color:
                indices.append(i)
        if len(indices) == 2:
            pos1, pos2 = indices[0], indices[1]
            if pos2 - pos1 == 2:
                count += 1
    print(count)

if __name__ == '__main__':
    solve()