n = int(input())
people = []
total_sum = 0
for _ in range(n):
    a, b = map(int, input().split())
    people.append((a, b))
    total_sum += b

if total_sum % 3 != 0:
    print(-1)
else:
    target = total_sum // 3
    sum_so_far = 0
    prev_dp = {(0, 0): 0}
    
    for a, b in people:
        current_dp = {}
        new_sum_so_far = sum_so_far + b
        for (s1, s2), changes in prev_dp.items():
            for t in [1, 2, 3]:
                new_s1 = s1 + (b if t == 1 else 0)
                new_s2 = s2 + (b if t == 2 else 0)
                sum3_new = new_sum_so_far - new_s1 - new_s2
                if new_s1 > target or new_s2 > target or sum3_new > target:
                    continue
                cost = changes + (1 if t != a else 0)
                key = (new_s1, new_s2)
                if key not in current_dp or cost < current_dp.get(key, float('inf')):
                    current_dp[key] = cost
        prev_dp = current_dp
        sum_so_far = new_sum_so_far
        if not prev_dp:
            break
    
    print(prev_dp.get((target, target), -1))