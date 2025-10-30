import bisect

n, T = input().split()
n = int(n)
len_T = len(T)

a_list = []
m_list = []

for _ in range(n):
    s = input().strip()
    # Compute a_i (max prefix)
    pos = 0
    for c in s:
        if pos < len_T and c == T[pos]:
            pos += 1
    a_i = pos
    # Compute m_i (min suffix)
    pos_m = len_T - 1
    for c in reversed(s):
        if pos_m >= 0 and c == T[pos_m]:
            pos_m -= 1
    m_i = pos_m + 1
    a_list.append(a_i)
    m_list.append(m_i)

a_list.sort()
total = 0

for m in m_list:
    idx = bisect.bisect_left(a_list, m)
    total += n - idx

print(total)