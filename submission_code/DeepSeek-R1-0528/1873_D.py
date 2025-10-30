t = int(input().strip())
for _ in range(t):
    n, k = map(int, input().split())
    s = input().strip()
    operations = 0
    last_cover = -1  # The last index covered by an operation
    for i in range(n):
        if s[i] == 'B':
            if i > last_cover:
                operations += 1
                last_cover = i + k - 1
    print(operations)