from collections import deque

def main():
    n = int(input())
    s = input().strip()
    t = input().strip()

    # Check if the counts of 'W' and 'B' are the same in s and t
    if s.count('W') != t.count('W') or s.count('B') != t.count('B'):
        print(-1)
        return

    # Construct initial and target states
    initial = s + '..'
    target = t + '..'

    if initial == target:
        print(0)
        return

    # BFS initialization
    visited = set()
    queue = deque()
    queue.append((initial, 0))
    visited.add(initial)

    while queue:
        current, steps = queue.popleft()

        # Find the position of the two consecutive '.'s
        i = -1
        for i in range(len(current) - 1):
            if current[i] == '.' and current[i+1] == '.':
                break
        else:
            # No two '.' found, which should not happen
            continue

        # Generate all possible next states
        for x in range(len(current) - 1):
            if current[x] != '.' and current[x+1] != '.':
                # Create a new state
                new_state = list(current)
                # Move stones from x and x+1 to the empty cells at i and i+1
                new_state[i] = current[x]
                new_state[i+1] = current[x+1]
                new_state[x] = '.'
                new_state[x+1] = '.'
                new_state_str = ''.join(new_state)

                if new_state_str == target:
                    print(steps + 1)
                    return

                if new_state_str not in visited:
                    visited.add(new_state_str)
                    queue.append((new_state_str, steps + 1))

    # If target not found
    print(-1)

if __name__ == "__main__":
    main()