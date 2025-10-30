n, t = map(int, input().split())
s = input().strip()
x = list(map(int, input().split()))

count = 0
for i in range(n):
    for j in range(i + 1, n):
        dir_i = 1 if s[i] == '1' else -1
        dir_j = 1 if s[j] == '1' else -1
        
        if dir_j == dir_i:
            continue  # Same direction, won't meet
        
        diff = x[i] - x[j]
        
        if dir_j - dir_i == 2:  # dir_j = 1, dir_i = -1
            if 0 <= diff <= 2 * t:
                count += 1
        else:  # dir_j - dir_i == -2, i.e., dir_j = -1, dir_i = 1
            if -2 * t <= diff <= 0:
                count += 1

print(count)