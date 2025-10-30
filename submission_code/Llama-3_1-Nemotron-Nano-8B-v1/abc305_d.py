import bisect

n = int(input())
A = list(map(int, input().split()))

s = []
e = []
for i in range((n - 1) // 2):
    s_i = A[2 * i + 1]
    e_i = A[2 * i + 2]
    s.append(s_i)
    e.append(e_i)

prefix_s = [0]
prefix_e = [0]
prefix_e_s = [0]
for i in range(len(s)):
    prefix_s.append(prefix_s[-1] + s[i])
    prefix_e.append(prefix_e[-1] + e[i])
    prefix_e_s.append(prefix_e_s[-1] + (e[i] - s[i]))

q = int(input())
for _ in range(q):
    l, r = map(int, input().split())
    
    # Find k: largest index with s[i] <= r
    k = bisect.bisect_right(s, r) - 1
    if k < 0:
        print(0)
        continue
    
    # Find m: first index in e[0..k] >= l
    m = bisect.bisect_left(e, l, 0, k + 1)
    if m > k:
        print(0)
        continue
    
    sum_e_s = prefix_e_s[k + 1] - prefix_e_s[m]
    
    # Find p: largest index in m..k with s[i] <= l
    p = bisect.bisect_right(s, l, m, k + 1) - 1
    if p < m:
        sum_L = 0
    else:
        sum_L = l * (p - m + 1) - (prefix_s[p + 1] - prefix_s[m])
    
    # Find q: first index in m..k with e[i] > r
    q_idx = bisect.bisect_right(e, r, m, k + 1)
    if q_idx > k:
        sum_R = 0
    else:
        sum_R = (prefix_e[k + 1] - prefix_e[q_idx]) - r * (k - q_idx + 1)
    
    total = sum_e_s - sum_L - sum_R
    print(max(0, total))