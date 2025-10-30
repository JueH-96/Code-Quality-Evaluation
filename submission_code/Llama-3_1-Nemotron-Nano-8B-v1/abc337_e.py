n = int(input())
m = (n - 1).bit_length()
print(m)
for i in range(m):
    subset = [x for x in range(1, n+1) if (x - 1) & (1 << i)]
    print(len(subset), end=' ')
    print(' '.join(map(str, subset)))
s = input().strip()
x = sum((1 << i) for i in range(m) if s[i] == '1') + 1
print(x)