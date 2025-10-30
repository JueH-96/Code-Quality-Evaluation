import bisect

n = int(input())
a = list(map(int, input().split()))
s = sorted(a)

pre_sum = [0]
current = 0
for num in s:
    current += num
    pre_sum.append(current)

result = []
for num in a:
    j = bisect.bisect_right(s, num)
    total = pre_sum[-1] - pre_sum[j]
    result.append(str(total))

print(' '.join(result))