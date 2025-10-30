import bisect

n = int(input())
A = list(map(int, input().split()))
k = (n - 1) // 2
s_list = []
e_list = []
for i in range(k):
    s = A[2 * i + 1]
    e = A[2 * i + 2]
    s_list.append(s)
    e_list.append(e)

sum_s = [0] * (k + 1)
sum_e = [0] * (k + 1)
for i in range(k):
    sum_s[i + 1] = sum_s[i] + s_list[i]
    sum_e[i + 1] = sum_e[i] + e_list[i]

q = int(input())
for _ in range(q):
    l, r = map(int, input().split())
    right_idx = bisect.bisect_right(s_list, r) - 1
    if right_idx < 0:
        print(0)
        continue
    left_idx = bisect.bisect_left(e_list, l, 0, right_idx + 1)
    if left_idx > right_idx:
        print(0)
        continue
    # Calculate sum_min_e
    j = bisect.bisect_right(e_list, r, left_idx, right_idx + 1) - 1
    sum_min_e = sum_e[j + 1] - sum_e[left_idx]
    if j < right_idx:
        sum_min_e += r * (right_idx - j)
    # Calculate sum_max_s
    k_idx = bisect.bisect_left(s_list, l, left_idx, right_idx + 1)
    sum_max_s = sum_s[right_idx + 1] - sum_s[k_idx]
    if k_idx > left_idx:
        sum_max_s += l * (k_idx - left_idx)
    print(sum_min_e - sum_max_s)