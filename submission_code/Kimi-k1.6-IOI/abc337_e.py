n = int(input())
m = (n - 1).bit_length()
friends = []
for j in range(m):
    lst = []
    for i in range(1, n + 1):
        if (i - 1) & (1 << j):
            lst.append(i)
    friends.append(lst)
print(m)
for lst in friends:
    print(len(lst), ' '.join(map(str, lst)))
s = input().strip()
x = 0
for j in range(m):
    x += int(s[j]) << j
print(x + 1)