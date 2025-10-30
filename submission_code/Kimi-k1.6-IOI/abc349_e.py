from functools import lru_cache

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    A = []
    for i in range(3):
        A.append(list(map(int, data[i*3 : (i+1)*3])))
    
    def has_t_win(state):
        for i in range(0, 9, 3):
            if state[i] == state[i+1] == state[i+2] == 1:
                return True
        for i in range(3):
            if state[i] == state[i+3] == state[i+6] == 1:
                return True
        if state[0] == state[4] == state[8] == 1:
            return True
        if state[2] == state[4] == state[6] == 1:
            return True
        return False
    
    def has_a_win(state):
        for i in range(0, 9, 3):
            if state[i] == state[i+1] == state[i+2] == 2:
                return True
        for i in range(3):
            if state[i] == state[i+3] == state[i+6] == 2:
                return True
        if state[0] == state[4] == state[8] == 2:
            return True
        if state[2] == state[4] == state[6] == 2:
            return True
        return False
    
    def calculate_sum(state):
        sum_t = 0
        sum_a = 0
        for i in range(3):
            for j in range(3):
                cell = state[i*3 + j]
                if cell == 1:
                    sum_t += A[i][j]
                elif cell == 2:
                    sum_a += A[i][j]
        return sum_t, sum_a
    
    @lru_cache(maxsize=None)
    def dfs(state):
        if has_t_win(state):
            return 'T'
        if has_a_win(state):
            return 'A'
        if all(cell != 0 for cell in state):
            sum_t, sum_a = calculate_sum(state)
            return 'T' if sum_t > sum_a else 'A'
        count_t = state.count(1)
        count_a = state.count(2)
        current_player = 'T' if count_t == count_a else 'A'
        possible_moves = [i for i, cell in enumerate(state) if cell == 0]
        if current_player == 'T':
            for move in possible_moves:
                new_state = list(state)
                new_state[move] = 1
                new_state = tuple(new_state)
                result = dfs(new_state)
                if result == 'T':
                    return 'T'
            return 'A'
        else:
            for move in possible_moves:
                new_state = list(state)
                new_state[move] = 2
                new_state = tuple(new_state)
                result = dfs(new_state)
                if result == 'A':
                    return 'A'
            return 'T'
    
    initial_state = (0,) * 9
    result = dfs(initial_state)
    print('Takahashi' if result == 'T' else 'Aoki')

if __name__ == "__main__":
    main()