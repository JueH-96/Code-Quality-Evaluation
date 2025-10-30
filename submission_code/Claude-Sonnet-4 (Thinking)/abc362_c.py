n = int(input())
intervals = []
for _ in range(n):
    l, r = map(int, input().split())
    intervals.append((l, r))

min_sum = sum(l for l, r in intervals)
max_sum = sum(r for l, r in intervals)

if min_sum > 0 or max_sum < 0:
    print("No")
else:
    print("Yes")
    result = [l for l, r in intervals]
    needed = 0 - min_sum
    
    for i in range(n):
        if needed == 0:
            break
        l, r = intervals[i]
        can_increase = r - l
        increase = min(needed, can_increase)
        result[i] += increase
        needed -= increase
    
    print(' '.join(map(str, result)))