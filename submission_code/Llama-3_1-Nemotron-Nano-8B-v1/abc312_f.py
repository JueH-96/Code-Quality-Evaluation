import bisect

n, m = map(int, input().split())

type0 = []
type1 = []
type2 = []

for _ in range(n):
    t, x = map(int, input().split())
    if t == 0:
        type0.append(x)
    elif t == 1:
        type1.append(x)
    else:
        type2.append(x)

# Sort and compute prefix sums for each type
type0.sort(reverse=True)
sum0 = [0]
for x in type0:
    sum0.append(sum0[-1] + x)

type1.sort(reverse=True)
sum1 = [0]
for x in type1:
    sum1.append(sum1[-1] + x)

type2.sort(reverse=True)
sum2 = [0]
for x in type2:
    sum2.append(sum2[-1] + x)

max_total = 0
max_s1 = min(m, len(type1))

for s1 in range(0, max_s1 + 1):
    required = s1
    # Determine minimal s2_min
    if not type2:
        if required > 0:
            continue
        s2_min = 0
    else:
        sum2_total = sum2[-1]
        if sum2_total < required:
            continue
        s2_min = bisect.bisect_left(sum2, required)
    
    if s1 + s2_min > m:
        continue
    
    remaining = m - s1 - s2_min
    k = min(remaining, len(type0))
    current_sum = sum1[s1] + sum0[k]
    
    if current_sum > max_total:
        max_total = current_sum

print(max_total)