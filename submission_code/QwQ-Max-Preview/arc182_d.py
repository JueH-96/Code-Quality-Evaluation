n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

possible_left_to_right = True
for i in range(n - 1):
    if b[i] == a[i + 1]:
        possible_left_to_right = False
        break

possible_right_to_left = True
for i in range(1, n):
    if b[i] == a[i - 1]:
        possible_right_to_left = False
        break

def is_in_clockwise_path(a_i, b_i, x):
    if b_i >= a_i:
        return a_i <= x <= b_i
    else:
        return x >= a_i or x <= b_i

def is_in_counter_path(a_i, b_i, x):
    if b_i <= a_i:
        return b_i <= x <= a_i
    else:
        return x <= a_i or x >= b_i

def compute_steps(a, b, direction):
    total = 0
    n_local = len(a)
    m_local = m
    for i in range(n_local):
        a_i = a[i]
        b_i = b[i]
        if a_i == b_i:
            continue
        steps_clockwise = (b_i - a_i) % m_local
        steps_counter = (a_i - b_i) % m_local
        allowed_clockwise = True
        allowed_counter = True
        
        if direction == 'ltr':
            B_prev = b[i-1] if i > 0 else None
            A_next = a[i+1] if i < n_local - 1 else None
            
            if steps_clockwise != 0:
                if i > 0 and is_in_clockwise_path(a_i, b_i, B_prev):
                    allowed_clockwise = False
                if i < n_local - 1 and is_in_clockwise_path(a_i, b_i, A_next):
                    allowed_clockwise = False
            if steps_counter != 0:
                if i > 0 and is_in_counter_path(a_i, b_i, B_prev):
                    allowed_counter = False
                if i < n_local - 1 and is_in_counter_path(a_i, b_i, A_next):
                    allowed_counter = False
        elif direction == 'rtl':
            A_prev = a[i-1] if i > 0 else None
            B_next = b[i+1] if i < n_local - 1 else None
            
            if steps_clockwise != 0:
                if i > 0 and is_in_clockwise_path(a_i, b_i, A_prev):
                    allowed_clockwise = False
                if i < n_local - 1 and is_in_clockwise_path(a_i, b_i, B_next):
                    allowed_clockwise = False
            if steps_counter != 0:
                if i > 0 and is_in_counter_path(a_i, b_i, A_prev):
                    allowed_counter = False
                if i < n_local - 1 and is_in_counter_path(a_i, b_i, B_next):
                    allowed_counter = False
        
        possible = []
        if allowed_clockwise:
            possible.append(steps_clockwise)
        if allowed_counter:
            possible.append(steps_counter)
        
        if not possible:
            return -1
        total += min(possible)
    return total

steps_left = -1
if possible_left_to_right:
    steps_left = compute_steps(a, b, 'ltr')

steps_right = -1
if possible_right_to_left:
    steps_right = compute_steps(a, b, 'rtl')

if steps_left == -1 and steps_right == -1:
    print(-1)
elif steps_left == -1:
    print(steps_right)
elif steps_right == -1:
    print(steps_left)
else:
    print(min(steps_left, steps_right))