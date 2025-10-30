# YOUR CODE HERE
import sys
import collections

def main():
    import sys
    import threading
    def solve():
        N = int(sys.stdin.readline())
        S = sys.stdin.readline().strip()
        T = sys.stdin.readline().strip()
        from collections import deque, defaultdict

        from itertools import combinations

        N2 = N + 2  # total positions

        initial_state = S + '..'
        target_state = T + '..'

        # Mapping: 'B' -> 0, 'W' -> 1, '.' -> 2
        symbol_to_number = {'B': 0, 'W': 1, '.': 2}
        number_to_symbol = {0: 'B', 1: 'W', 2: '.'}

        def state_to_int(state):
            num = 0
            for c in state:
                num = num * 3 + symbol_to_number[c]
            return num

        def int_to_state(num):
            state = ''
            for _ in range(N2):
                num, r = divmod(num, 3)
                state = number_to_symbol[r] + state
            return state

        initial_num = state_to_int(initial_state)
        target_num = state_to_int(target_state)

        from collections import deque

        visited = {}
        queue = deque()
        queue.append((initial_num, 0))
        visited[initial_num] = 0

        while queue:
            curr_num, moves = queue.popleft()
            if curr_num == target_num:
                print(moves)
                return
            curr_state = int_to_state(curr_num)
            empty_positions = [i for i, c in enumerate(curr_state) if c == '.']
            if len(empty_positions) != 2:
                continue  # Should not happen
            e1, e2 = empty_positions

            for x in range(N2 -1):  # x from 0 to N2 -2
                # Check if positions x and x+1 are occupied
                if curr_state[x] != '.' and curr_state[x+1] != '.':
                    # Swap stones at x and x+1 with empty positions e1 and e2
                    new_state = list(curr_state)
                    # Move stones to empty positions, preserving order
                    new_state[e1] = curr_state[x]
                    new_state[e2] = curr_state[x+1]
                    # Positions x and x+1 become empty
                    new_state[x] = '.'
                    new_state[x+1] = '.'
                    new_num = state_to_int(''.join(new_state))
                    if new_num not in visited:
                        visited[new_num] = moves +1
                        queue.append((new_num, moves+1))

        # If we reach here, it's impossible
        print(-1)
    threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()