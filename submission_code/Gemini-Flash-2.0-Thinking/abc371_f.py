def solve():
    n = int(input())
    x = list(map(int, input().split()))
    q = int(input())
    tasks = []
    for _ in range(q):
        tasks.append(list(map(int, input().split())))

    current_positions = list(x)
    total_movements = 0

    for t_index, g_val in tasks:
        person_index = t_index - 1
        target_position = g_val
        current_pos = current_positions[person_index]

        movements = abs(current_pos - target_position)
        total_movements += movements
        current_positions[person_index] = target_position

        # Sort the positions to maintain the order
        sorted_positions = sorted(current_positions)
        current_positions = list(sorted_positions)

    # Recalculate movements based on the intended interpretation

    current_positions = list(x)
    total_movements = 0

    for t_index, g_val in tasks:
        person_index = t_index - 1
        target_position = g_val

        initial_positions = list(current_positions)
        target_positions = list(current_positions)
        target_positions[person_index] = target_position
        target_positions.sort()

        task_movements = 0
        for i in range(n):
            task_movements += abs(target_positions[i] - initial_positions[i])

        total_movements += task_movements
        current_positions = list(target_positions)

    print(total_movements)

solve()