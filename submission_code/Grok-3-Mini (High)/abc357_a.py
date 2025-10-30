import sys
data = sys.stdin.read().split()
idx = 0
N = int(data[idx])
idx += 1
M = int(data[idx])
idx += 1
H_list = [int(data[j]) for j in range(idx, idx + N)]
count = 0
remaining = M
for h in H_list:
    if remaining >= h:
        count += 1
        remaining -= h
    else:
        remaining = 0
        break  # No need to check further aliens as remaining is 0
print(count)