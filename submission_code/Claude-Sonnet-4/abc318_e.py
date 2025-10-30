n = int(input())
a = list(map(int, input().split()))

count = 0

for i in range(n):
    for k in range(i+2, n):
        if a[i] == a[k]:
            for j in range(i+1, k):
                if a[j] != a[i]:
                    count += 1

print(count)