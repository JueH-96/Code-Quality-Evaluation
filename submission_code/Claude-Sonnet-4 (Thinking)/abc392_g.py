n = int(input())
s = list(map(int, input().split()))

s_set = set(s)
s.sort()
count = 0

for i in range(n):
    for j in range(i+1, n):
        a, c = s[i], s[j]
        if (a + c) % 2 == 0:  # B must be an integer
            b = (a + c) // 2
            if b in s_set:
                count += 1

print(count)