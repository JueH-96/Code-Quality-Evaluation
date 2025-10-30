M = int(input())
S1 = input().strip()
S2 = input().strip()
S3 = input().strip()

from collections import defaultdict

def generate_times(S, M):
    times = defaultdict(set)
    for k in range(M):
        d = S[k]
        x = 0
        while True:
            t = k + x * M
            if t > 1000:  # Arbitrary limit to prevent infinite loops
                break
            times[d].add(t)
            x += 1
    for d in times:
        times[d] = sorted(times[d])
    return times

reel1 = generate_times(S1, M)
reel2 = generate_times(S2, M)
reel3 = generate_times(S3, M)

min_time = float('inf')

for d in '0123456789':
    if d not in reel1 or d not in reel2 or d not in reel3:
        continue
    t1_list = reel1[d]
    t2_list = reel2[d]
    t3_list = reel3[d]
    for t1 in t1_list:
        for t2 in t2_list:
            if t2 == t1:
                continue
            for t3 in t3_list:
                if t3 == t1 or t3 == t2:
                    continue
                current_max = max(t1, t2, t3)
                if current_max < min_time:
                    min_time = current_max

if min_time != float('inf'):
    print(min_time)
else:
    print(-1)