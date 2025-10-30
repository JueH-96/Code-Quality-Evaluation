n, q = map(int, input().split())
l = 1
r = 2
total = 0

for _ in range(q):
    h, t = input().split()
    target = int(t)
    if h == 'L':
        current = l
        other_pos = r
    else:
        current = r
        other_pos = l

    steps_clockwise = (target - current) % n
    valid1 = False
    if steps_clockwise != 0:
        path1 = []
        pos = current
        while len(path1) < steps_clockwise:
            path1.append(pos)
            pos = (pos % n) + 1
            if pos > n:
                pos = 1
        if other_pos not in path1:
            valid1 = True

    steps_counter = (current - target) % n
    valid2 = False
    if steps_counter != 0:
        path2 = []
        pos = current
        while len(path2) < steps_counter:
            path2.append(pos)
            pos = (pos % n) - 1
            if pos < 1:
                pos = n
        if other_pos not in path2:
            valid2 = True

    valid_steps = []
    if valid1:
        valid_steps.append(steps_clockwise)
    if valid2:
        valid_steps.append(steps_counter)

    min_steps = min(valid_steps)
    total += min_steps

    if h == 'L':
        l = target
    else:
        r = target

print(total)