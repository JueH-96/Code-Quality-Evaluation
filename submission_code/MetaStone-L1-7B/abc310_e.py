n = int(input())
s = input().strip()

cnt0 = 0
cnt1 = 0
total = 0

for c in s:
    if c == '0':
        new_cnt0 = 1
        new_cnt1 = cnt0 + cnt1
    else:
        new_cnt0 = cnt1
        new_cnt1 = cnt0 + 1
    total += new_cnt1
    cnt0, cnt1 = new_cnt0, new_cnt1

print(total)