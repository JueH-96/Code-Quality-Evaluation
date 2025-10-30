from collections import deque

def longestString(x: int, y: int, z: int) -> int:
    max_strings = 0
    initial_state = (x, y, z, None, 0)
    queue = deque()
    visited = {}
    queue.append(initial_state)
    visited[initial_state] = 0

    while queue:
        current = queue.popleft()
        current_x, current_y, current_z, last_char, run_length = current
        current_strings = visited[current]

        # Try adding AA
        if current_x >= 1:
            new_x = current_x - 1
            new_y = current_y
            new_z = current_z
            s_char = 'A'
            e_char = 'A'

            if last_char == 'A':
                new_r = run_length + 1
                if new_r > 2:
                    continue
            else:
                new_r = 1

            new_state = (new_x, new_y, new_z, e_char, new_r)
            if new_state not in visited or (current_strings + 1) > visited.get(new_state, -1):
                visited[new_state] = current_strings + 1
                queue.append(new_state)

        # Try adding BB
        if current_y >= 1:
            new_x = current_x
            new_y = current_y - 1
            new_z = current_z
            s_char = 'B'
            e_char = 'B'

            if last_char == 'B':
                new_r = run_length + 1
                if new_r > 2:
                    continue
            else:
                new_r = 1

            new_state = (new_x, new_y, new_z, e_char, new_r)
            if new_state not in visited or (current_strings + 1) > visited.get(new_state, -1):
                visited[new_state] = current_strings + 1
                queue.append(new_state)

        # Try adding AB
        if current_z >= 1:
            new_x = current_x
            new_y = current_y
            new_z = current_z - 1
            s_char = 'A'
            e_char = 'B'

            if last_char == 'A':
                new_r = run_length + 1
                if new_r > 2:
                    continue
            else:
                new_r = 1

            new_state = (new_x, new_y, new_z, e_char, new_r)
            if new_state not in visited or (current_strings + 1) > visited.get(new_state, -1):
                visited[new_state] = current_strings + 1
                queue.append(new_state)

    max_length = 2 * max_strings
    return max_length