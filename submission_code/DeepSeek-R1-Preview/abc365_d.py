def beats(t, a):
    return (t == 'R' and a == 'S') or (t == 'S' and a == 'P') or (t == 'P' and a == 'R')

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1].strip()
    
    if N == 0:
        print(0)
        return
    
    allowed_moves_list = []
    for c in S:
        if c == 'R':
            allowed_moves_list.append(['R', 'P'])
        elif c == 'P':
            allowed_moves_list.append(['P', 'S'])
        else:
            allowed_moves_list.append(['S', 'R'])
    
    current_dp = {}
    for move in allowed_moves_list[0]:
        current_dp[move] = 1 if beats(move, S[0]) else 0
    
    for i in range(1, N):
        allowed_prev = allowed_moves_list[i-1]
        allowed_current = allowed_moves_list[i]
        new_dp = {}
        for m in allowed_current:
            possible_prev = [pm for pm in allowed_prev if pm != m]
            best_prev = max(current_dp[pm] for pm in possible_prev)
            add = 1 if beats(m, S[i]) else 0
            new_dp[m] = best_prev + add
        current_dp = new_dp
    
    print(max(current_dp.values()))

if __name__ == '__main__':
    main()