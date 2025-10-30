N, Q = map(int, input().split())
left = 1
right = 2
total = 0

for _ in range(Q):
    H, T = input().split()
    T = int(T)
    if H == 'L':
        l = left
        r = right
        target = T
        cw = (target - l) % N
        if (r - l) % N <= cw:
            steps = (l - target) % N
        else:
            steps = cw
        total += steps
        left = target
    else:
        r_curr = right
        l_curr = left
        target = T
        cw = (target - r_curr) % N
        if (l_curr - r_curr) % N <= cw:
            steps = (r_curr - target) % N
        else:
            steps = cw
        total += steps
        right = target

print(total)