import bisect

n, T = input().split()
n = int(n)
T = T.strip()
len_T = len(T)
S_list = [input().strip() for _ in range(n)]

# Compute k_i for each string
k_i = []
for S in S_list:
    pos = 0
    for c in S:
        if pos < len_T and c == T[pos]:
            pos += 1
    k_i.append(pos)

# Compute m_j for each string
m_j = []
for S in S_list:
    pos = len_T - 1
    for c in reversed(S):
        if pos >= 0 and c == T[pos]:
            pos -= 1
    m_j.append(pos + 1)

# Sort k_i for binary search
sorted_k = sorted(k_i)

# Calculate total pairs
total = 0
for m in m_j:
    idx = bisect.bisect_left(sorted_k, m)
    total += len(sorted_k) - idx

print(total)