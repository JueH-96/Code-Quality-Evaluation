import bisect
import math

n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Create sorted list of (A_i, original index)
sorted_pairs = sorted((a, i) for i, a in enumerate(A))
sorted_values = [p[0] for p in sorted_pairs]
original_indices = [p[1] for p in sorted_pairs]

# Build sparse table for range minimum query
max_log = 0
if n > 0:
    max_log = math.floor(math.log2(n))
st = []
st.append(original_indices.copy())
for j in range(1, max_log + 1):
    prev_st = st[j-1]
    current_st = []
    for i in range(n - (1 << j) + 1):
        current_st.append(min(prev_st[i], prev_st[i + (1 << (j-1))]))
    st.append(current_st)

def get_min(l, r):
    length = r - l + 1
    if length == 0:
        return -1
    k = length.bit_length() - 1
    if (1 << k) > length:
        k -= 1
    right = r - (1 << k) + 1
    if right < l:
        return st[k][l]
    return min(st[k][l], st[k][right])

for b in B:
    # Find the largest index k where sorted_values[k] <= b
    k = bisect.bisect_right(sorted_values, b) - 1
    if k < 0:
        print(-1)
    else:
        min_idx = get_min(0, k)
        print(min_idx + 1)