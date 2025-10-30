import sys
data = list(map(int, sys.stdin.read().split()))
index = 0
N = data[index]
index += 1
A = data[index]
index += 1
B = data[index]
index += 1
P = A + B
residues = set(d % P for d in data[index:index + N])
intervals = []
for r in residues:
    S_bad = (A - r) % P
    E_linear = S_bad + B - 1
    if E_linear < P:
        intervals.append((S_bad, E_linear))
    else:
        intervals.append((S_bad, P - 1))
        intervals.append((0, E_linear - P))
intervals.sort()
if not intervals:
    print("Yes")
else:
    merged = []
    curr_start, curr_end = intervals[0]
    for i in range(1, len(intervals)):
        next_start, next_end = intervals[i]
        if next_start <= curr_end + 1:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start = next_start
            curr_end = next_end
    merged.append((curr_start, curr_end))
    if len(merged) == 1 and merged[0][0] == 0 and merged[0][1] == P - 1:
        print("No")
    else:
        print("Yes")