lines = [
    (0,1,2), (3,4,5), (6,7,8),
    (0,3,6), (1,4,7), (2,5,8),
    (0,4,8), (2,4,6)
]

def main():
    grid_flat = []
    for _ in range(3):
        row = list(map(int, input().split()))
        grid_flat.extend(row)
    
    memo = {}
    
    def dfs(state):
        if state in memo:
            return memo[state]
        
        moves_done = 9 - state.count(0)
        winner = None
        
        for line in lines:
            a, b, c = line
            if state[a] != 0 and state[a] == state[b] == state[c]:
                if state[a] == 1:
                    winner = 1
                else:
                    winner = -1
                break
        
        if winner is not None:
            memo[state] = winner
            return winner
        
        if moves_done == 9:
            s1 = 0
            s2 = 0
            for i in range(9):
                if state[i] == 1:
                    s1 += grid_flat[i]
                elif state[i] == 2:
                    s2 += grid_flat[i]
            if s1 > s2:
                winner = 1
            else:
                winner = -1
            memo[state] = winner
            return winner
        
        next_player = 1 if (moves_done % 2 == 0) else 2
        available_moves = [i for i in range(9) if state[i] == 0]
        
        if next_player == 1:
            best = -10**20
            for move in available_moves:
                lst = list(state)
                lst[move] = 1
                new_state = tuple(lst)
                res = dfs(new_state)
                if res > best:
                    best = res
                if best == 1:
                    break
            memo[state] = best
            return best
        else:
            best = 10**20
            for move in available_moves:
                lst = list(state)
                lst[move] = 2
                new_state = tuple(lst)
                res = dfs(new_state)
                if res < best:
                    best = res
                if best == -1:
                    break
            memo[state] = best
            return best
    
    initial_state = (0,0,0,0,0,0,0,0,0)
    result = dfs(initial_state)
    if result == 1:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()