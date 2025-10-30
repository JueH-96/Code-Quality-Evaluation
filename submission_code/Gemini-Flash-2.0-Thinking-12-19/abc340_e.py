def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for op_idx in range(m):
        current_b = b[op_idx]
        balls_in_hand = a[current_b]
        a[current_b] = 0
        counts = [0] * n
        for c in range(1, balls_in_hand + 1):
            box_index = (current_b + c) % n
            counts[box_index] += 1
        for j in range(n):
            a[j] += counts[j]

    print(*(a))

solve()