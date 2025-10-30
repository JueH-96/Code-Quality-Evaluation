def solve():
    n = int(input())
    a = list(map(int, input().split()))
    results = []

    for k in range(1, n + 1):
        start_index = k - 1
        max_size = a[start_index]

        q = [(a[start_index], list(a))]
        visited = set()

        while q:
            current_takahashi_size, current_arrangement = q.pop(0)

            state_tuple = (current_takahashi_size, tuple(current_arrangement))
            if state_tuple in visited:
                continue
            visited.add(state_tuple)

            max_size = max(max_size, current_takahashi_size)

            if len(current_arrangement) == 1:
                continue

            takahashi_index = -1
            for i, size in enumerate(current_arrangement):
                if size == current_takahashi_size:
                    takahashi_index = i
                    break

            if takahashi_index == -1:
                continue

            # Try absorbing left
            if takahashi_index > 0 and current_arrangement[takahashi_index - 1] < current_takahashi_size:
                new_arrangement = list(current_arrangement)
                absorbed_size = new_arrangement.pop(takahashi_index - 1)
                q.append((current_takahashi_size + absorbed_size, new_arrangement))

            # Try absorbing right
            if takahashi_index < len(current_arrangement) - 1 and current_arrangement[takahashi_index + 1] < current_takahashi_size:
                new_arrangement = list(current_arrangement)
                absorbed_size = new_arrangement.pop(takahashi_index + 1)
                q.append((current_takahashi_size + absorbed_size, new_arrangement))

        results.append(max_size)

    print(*results)

solve()