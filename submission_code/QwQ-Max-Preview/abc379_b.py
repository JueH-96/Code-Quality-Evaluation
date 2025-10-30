n, k = map(int, input().split())
s = input().strip()
count = 0
i = 0
while i <= n - k:
    if s[i:i+k] == 'O' * k:
        count += 1
        i += k
    else:
        i += 1
print(count)