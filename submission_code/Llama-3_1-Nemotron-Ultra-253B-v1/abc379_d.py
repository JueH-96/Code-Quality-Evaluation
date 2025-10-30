import bisect

Q = int(input())
current_time = 0
births = []
start = 0

for _ in range(Q):
    parts = input().split()
    if parts[0] == '1':
        births.append(current_time)
    elif parts[0] == '2':
        T = int(parts[1])
        current_time += T
    else:
        H = int(parts[1])
        target = current_time - H
        pos = bisect.bisect_right(births, target, start)
        count = pos - start
        print(count)
        start = pos