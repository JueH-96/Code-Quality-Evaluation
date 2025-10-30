n = int(input())
a = list(map(int, input().split()))

pair_list = []
i = 0
while i < len(a):
    if i + 1 < len(a) and a[i] == a[i + 1]:
        pair_list.append(a[i])
        i += 2
    else:
        i += 1

max_len = 0
last_occurrence = {}
left = 0

for right in range(len(pair_list)):
    num = pair_list[right]
    if num in last_occurrence and last_occurrence[num] >= left:
        left = last_occurrence[num] + 1
    last_occurrence[num] = right
    current_len = right - left + 1
    if current_len > max_len:
        max_len = current_len

print(max_len * 2)