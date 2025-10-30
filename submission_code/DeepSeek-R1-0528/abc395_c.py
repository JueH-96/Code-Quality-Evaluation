n = int(input().strip())
arr = list(map(int, input().split()))
last_occurrence = {}
min_ans = 10**9

for i, num in enumerate(arr):
    if num in last_occurrence:
        seg_len = i - last_occurrence[num] + 1
        if seg_len < min_ans:
            min_ans = seg_len
            if min_ans == 2:
                break
        last_occurrence[num] = i
    else:
        last_occurrence[num] = i

print(min_ans if min_ans != 10**9 else -1)