def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for op_idx in range(m):
        box_index = b[op_idx]
        balls_in_hand = a[box_index]
        a[box_index] = 0

        full_cycles = balls_in_hand // n
        remaining_balls = balls_in_hand % n

        for i in range(n):
            a[i] += full_cycles

        for i in range(1, remaining_balls + 1):
            dist_box_index = (box_index + i) % n
            a[dist_box_index] += 1

    print(*(a))

if __name__ == '__main__':
    solve()