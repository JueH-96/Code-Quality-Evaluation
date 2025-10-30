n, m = map(int, input().split())

R = set()
C = set()
D1 = set()
D2 = set()

for _ in range(m):
    a, b = map(int, input().split())
    R.add(a)
    C.add(b)
    D1.add(a + b)
    D2.add(a - b)

a_val = len(R) * n
b_val = len(C) * n

sum_c = 0
for s in D1:
    temp = n - abs(s - (n + 1))
    sum_c += temp

sum_d = 0
for d in D2:
    sum_d += (n - abs(d))

ab = len(R) * len(C)

ac = 0
for r in R:
    for s in D1:
        j = s - r
        if 1 <= j <= n:
            ac += 1

ad = 0
for r in R:
    for d in D2:
        j = r - d
        if 1 <= j <= n:
            ad += 1

bc = 0
for c in C:
    for s in D1:
        i = s - c
        if 1 <= i <= n:
            bc += 1

bd = 0
for c in C:
    for d in D2:
        i = c + d
        if 1 <= i <= n:
            bd += 1

cd = 0
for s in D1:
    for d in D2:
        if (s + d) % 2 != 0:
            continue
        i = (s + d) // 2
        j = (s - d) // 2
        if 1 <= i <= n and 1 <= j <= n:
            cd += 1

abc = 0
for r in R:
    for c in C:
        if (r + c) in D1:
            abc += 1

abd = 0
for r in R:
    for c in C:
        if (r - c) in D2:
            abd += 1

acd = 0
for r in R:
    for s in D1:
        j = s - r
        if 1 <= j <= n:
            current_d = r - j
            if current_d in D2:
                acd += 1

bcd = 0
for c in C:
    for s in D1:
        d_required = s - 2 * c
        if d_required in D2:
            bcd += 1

abcd = 0
for r in R:
    for c in C:
        s = r + c
        d = r - c
        if s in D1 and d in D2:
            abcd += 1

forbidden = (a_val + b_val + sum_c + sum_d) - (ab + ac + ad + bc + bd + cd) + (abc + abd + acd + bcd) - abcd
answer = n * n - forbidden

print(answer)