n = int(input())
s = input().strip()

sum_total = 0
prev0 = 0
prev1 = 0

for c in s:
    a = int(c)
    if a == 0:
        ext1 = prev0 + prev1
        ext0 = 0
        single0 = 1
        single1 = 0
    else:
        ext1 = prev0
        ext0 = prev1
        single0 = 0
        single1 = 1
    new0 = ext0 + single0
    new1 = ext1 + single1
    sum_total += new1
    prev0, prev1 = new0, new1

print(sum_total)