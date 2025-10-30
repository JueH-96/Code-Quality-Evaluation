n = int(input())
h = list(map(int, input().split()))
max_count = 1

for i in range(n):
    for j in range(i + 1, n):
        if h[i] != h[j]:
            continue
        s = j - i
        current = j + s
        count = 2
        while current < n:
            if h[current] == h[i]:
                count += 1
                current += s
            else:
                break
        if count > max_count:
            max_count = count

print(max_count)