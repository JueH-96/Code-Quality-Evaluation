from collections import deque

def min_operations():
    N = int(input())
    S = input()
    T = input()

    def get_state(s):
        return [(i, c) for i, c in enumerate(s)]

    def get_next_states(state):
        next_states = []
        for i in range(1, N):
            if state[i-1][1] == state[i][1] == 'B' or state[i-1][1] == state[i][1] == 'W':
                new_state = state[:i-1] + [state[i-1], state[i]] + state[i+2:]
                next_states.append(new_state)
        return next_states

    def get_distance(start, end):
        queue = deque([(start, 0)])
        visited = set([start])
        while queue:
            state, dist = queue.popleft()
            if state == end:
                return dist
            for next_state in get_next_states(state):
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, dist+1))
        return -1

    start_state = get_state(S)
    end_state = get_state(T)
    print(get_distance(start_state, end_state))

min_operations()