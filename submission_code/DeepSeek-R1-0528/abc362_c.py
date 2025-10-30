n = int(input().strip())
L = []
R = []
total_min = 0
total_max = 0

for _ in range(n):
    a, b = map(int, input().split())
    L.append(a)
    R.append(b)
    total_min += a
    total_max += b

if total_min <= 0 <= total_max:
    D = -total_min
    rem = D
    ans = []
    for i in range(n):
        if rem > 0:
            a_i = R[i] - L[i]
            take = min(rem, a_i)
            ans.append(str(L[i] + take))
            rem -= take
        else:
            ans.append(str(L[i]))
    print("Yes")
    print(" ".join(ans))
else:
    print("No")