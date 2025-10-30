n = int(input())
s = input().strip()
sorted_s = sorted(s)

count = 0

if n == 1:
    # Handle N=1 case separately
    for a in range(4):
        x = a * a
        s_x = str(x)
        if sorted(s_x) == sorted_s:
            count += 1
else:
    a_max = int((10**n - 1) ** 0.5)
    for a in range(a_max + 1):
        x = a * a
        s_x = str(x).zfill(n)
        if sorted(s_x) == sorted_s:
            count += 1

print(count)