n = int(input())
A = list(map(int, input().split()))

pair_list = []
for i in range(n - 1):
    if A[i] == A[i + 1]:
        pair_list.append(A[i])

if not pair_list:
    print(0)
else:
    max_len = 0
    left = 0
    last_occurrence = {}
    for right in range(len(pair_list)):
        num = pair_list[right]
        if num in last_occurrence and last_occurrence[num] >= left:
            left = last_occurrence[num] + 1
        last_occurrence[num] = right
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    print(max_len * 2)