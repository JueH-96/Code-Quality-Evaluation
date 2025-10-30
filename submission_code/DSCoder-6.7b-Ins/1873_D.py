def min_operations(n, k, s):
    operations = 0
    i = 0
    while i < n:
        if s[i] == 'B':
            start = i
            while i < n and s[i] == 'B':
                i += 1
            end = i
            operations += (end - start) // k
            if (end - start) % k != 0:
                operations += 1
        else:
            i += 1
    return operations

t = int(input().strip())
for _ in range(t):
    n, k = map(int, input().strip().split())
    s = input().strip()
    print(min_operations(n, k, s))