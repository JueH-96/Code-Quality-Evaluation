n = int(input())
s = input().strip()

if n == 0:
    print(0)
    exit()

sum_total = 0

a = s[0] == '1'
prev0 = 1 - a
prev1 = a
sum_total += prev1

for j in range(1, n):
    current_a = s[j] == '1'
    current1_ext = prev0 + (prev1 if not current_a else 0)
    current1_new = 1 if current_a else 0
    cnt1 = current1_ext + current1_new
    current0_ext = prev1 if current_a else 0
    current0_new = 1 if not current_a else 0
    cnt0 = current0_ext + current0_new
    sum_total += cnt1
    prev0, prev1 = cnt0, cnt1

print(sum_total)