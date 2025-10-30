from collections import deque

def solve():
    n = int(input())
    s = input()
    t = input()

    start_state = s + ".."
    target_state = t + ".."

    if s.count('W') != t.count('W') or s.count('B') != t.count('B'):
        print(-1)
        return

    queue = deque([(start_state, 0)])
    visited = {start_state}

    while queue:
        current_state, num_ops = queue.popleft()

        if current_state == target_state:
            print(num_ops)
            return

        empty_indices = [i for i, char in enumerate(current_state) if char == '.']
        if len(empty_indices) != 2 or empty_indices[1] != empty_indices[0] + 1:
            continue

        adj_empty = tuple(empty_indices)

        stones = [i for i, char in enumerate(current_state) if char != '.']

        for i in range(len(stones) - 1):
            stone1_idx = stones[i]
            stone2_idx = stones[i+1]

            if stone2_idx == stone1_idx + 1:
                new_state = perform_move(current_state, (stone1_idx, stone2_idx), adj_empty)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, num_ops + 1))

    print(-1)

def perform_move(state, stone_indices, empty_indices):
    state_list = list(state)
    s1, s2 = stone_indices
    e1, e2 = empty_indices

    moved_stones = state_list[s1:s2+1]

    new_state_list = list(state)

    new_state_list[s1] = '.'
    new_state_list[s2] = '.'
    new_state_list[e1] = moved_stones[0]
    new_state_list[e2] = moved_stones[1]

    return "".join(new_state_list)

if __name__ == "__main__":
    solve()