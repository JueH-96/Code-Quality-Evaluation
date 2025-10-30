n = int(input())
a = list(map(int, input().split()))

if n == 1:
    print(1)
else:
    d = [a[i+1] - a[i] for i in range(n-1)]
    total = 0
    current = d[0]
    count = 1
    for i in range(1, len(d)):
        if d[i] == current:
            count += 1
        else:
            total += count * (count + 1) // 2
            current = d[i]
            count = 1
    total += count * (count + 1) // 2
    total += n
    print(total)