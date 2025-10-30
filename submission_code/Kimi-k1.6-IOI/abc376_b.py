n, q = map(int, input().split())
instructions = [input().split() for _ in range(q)]

l = 1
r = 2
total = 0

def is_in_clockwise(A, B, C, N):
    if A < B:
        return A <= C <= B
    else:
        return C >= A or C <= B

def min_steps(A, B, C, N):
    if A == B:
        return 0
    d_clock = (B - A) % N
    d_counter = (A - B) % N
    clock_path = is_in_clockwise(A, B, C, N)
    counter_path = is_in_clockwise(B, A, C, N)
    options = []
    if not clock_path:
        options.append(d_clock)
    if not counter_path:
        options.append(d_counter)
    return min(options)

for h, t_str in instructions:
    t = int(t_str)
    if h == 'L':
        a, b, c = l, t, r
    else:
        a, b, c = r, t, l
    steps = min_steps(a, b, c, n)
    total += steps
    if h == 'L':
        l = b
    else:
        r = b

print(total)