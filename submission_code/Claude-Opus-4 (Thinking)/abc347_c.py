def interval_intersection(i1, i2):
    a1, b1 = i1
    a2, b2 = i2
    a = max(a1, a2)
    b = min(b1, b2)
    if a <= b:
        return (a, b)
    else:
        return None

def set_intersection(s1, s2):
    result = []
    for i1 in s1:
        for i2 in s2:
            inter = interval_intersection(i1, i2)
            if inter:
                result.append(inter)
    return result

N, A, B = map(int, input().split())
D = list(map(int, input().split()))

P = A + B

sets = []
for d in D:
    r = d % P
    if r < A:
        intervals = [(0, A - r - 1)]
        if r > 0:
            intervals.append((P - r, P - 1))
        sets.append(intervals)
    else:
        sets.append([(P - r, P - r + A - 1)])

result = sets[0]
for i in range(1, N):
    result = set_intersection(result, sets[i])
    if not result:
        break

if result:
    print("Yes")
else:
    print("No")