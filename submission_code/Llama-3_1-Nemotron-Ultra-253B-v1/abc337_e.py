n = int(input())
m = (n - 1).bit_length()
print(m)
friends = [[] for _ in range(m)]
for x in range(1, n + 1):
    for i in range(1, m + 1):
        mask = 1 << (m - i)
        if (x - 1) & mask:
            friends[i - 1].append(x)
for lst in friends:
    print(len(lst), end=' ')
    print(' '.join(map(str, lst)))
s = input().strip()
x_prime = int(s, 2) + 1
print(x_prime)