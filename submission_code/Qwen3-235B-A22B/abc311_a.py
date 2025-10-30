n = int(input())
s = input().strip()

count_a = 0
count_b = 0
count_c = 0

for i in range(n):
    char = s[i]
    if char == 'A':
        count_a += 1
    elif char == 'B':
        count_b += 1
    else:
        count_c += 1
    if count_a >= 1 and count_b >= 1 and count_c >= 1:
        print(i + 1)
        break