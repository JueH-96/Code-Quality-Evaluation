import bisect

n = int(input())
A = list(map(int, input().split()))
S = sorted(A)
prefix = [0]
current_sum = 0
for num in S:
    current_sum += num
    prefix.append(current_sum)
total_sum = prefix[-1]
result = []
for a in A:
    j = bisect.bisect_right(S, a)
    result.append(str(total_sum - prefix[j]))
print(' '.join(result))