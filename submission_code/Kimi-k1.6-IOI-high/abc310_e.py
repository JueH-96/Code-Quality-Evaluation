n = int(input())
s = input().strip()

count0 = 0
count1 = 0
total = 0

for c in s:
    if c == '0':
        new_count0 = 1
        new_count1 = count0 + count1
    else:
        new_count0 = count1
        new_count1 = count0 + 1
    total += new_count1
    count0, count1 = new_count0, new_count1

print(total)