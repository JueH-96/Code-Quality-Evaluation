n = int(input())
H = list(map(int, input().split()))
max_count = 1

for i in range(n):
    current_h = H[i]
    max_d = (n - 1 - i)
    for d in range(1, max_d + 1):
        count = 1
        next_pos = i + d
        while next_pos < n and H[next_pos] == current_h:
            count += 1
            next_pos += d
        if count > max_count:
            max_count = count

print(max_count)