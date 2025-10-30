import bisect
from collections import defaultdict

n = int(input())
A = list(map(int, input().split()))
S = input().strip()

M_pos = []
E_pos = []
X_pos = []
for i in range(n):
    c = S[i]
    if c == 'M':
        M_pos.append(i)
    elif c == 'E':
        E_pos.append(i)
    elif c == 'X':
        X_pos.append(i)

ma_dict = defaultdict(list)
eb_dict = defaultdict(list)
xc_dict = defaultdict(list)

for m in M_pos:
    ma_dict[A[m]].append(m)
for e in E_pos:
    eb_dict[A[e]].append(e)
for x in X_pos:
    xc_dict[A[x]].append(x)

total = 0

for a in [0, 1, 2]:
    for b in [0, 1, 2]:
        for c in [0, 1, 2]:
            ma = ma_dict.get(a, [])
            eb = eb_dict.get(b, [])
            xc = xc_dict.get(c, [])
            if not ma or not eb or not xc:
                continue
            
            sum_x = [0] * (len(eb) + 1)
            for i in reversed(range(len(eb))):
                e = eb[i]
                pos = bisect.bisect_right(xc, e)
                x_count = len(xc) - pos
                sum_x[i] = sum_x[i + 1] + x_count
            
            current_count = 0
            for m in ma:
                idx = bisect.bisect_right(eb, m)
                current_count += sum_x[idx]
            
            s = {a, b, c}
            if 0 not in s:
                mex_val = 0
            elif 1 not in s:
                mex_val = 1
            elif 2 not in s:
                mex_val = 2
            else:
                mex_val = 3
            total += current_count * mex_val

print(total)