# YOUR CODE HERE

t = int(input().strip())

for _ in range(t):
    n, k = map(int, input().strip().split())
    s = input().strip()

    operations = 0
    i = 0

    while i < n:
        group_end = i
        while group_end < n and s[group_end] == 'W':
            group_end += 1

        if group_end - i >= k:
            operations += (group_end - i) // k
            i = group_end
        else:
            i = group_end

    print(operations)