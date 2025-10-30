# YOUR CODE HERE
def solve():
    n = int(input())
    s = input()
    
    ans = 0
    for start in ['R', 'P', 'S']:
        wins = 0
        moves = []
        
        for i in range(n):
            if start == 'R':
                if s[i] == 'S':
                    wins += 1
                    moves.append('R')
                elif s[i] == 'R':
                    moves.append('R')
                else:
                    moves.append('S')
            elif start == 'P':
                if s[i] == 'R':
                    wins += 1
                    moves.append('P')
                elif s[i] == 'P':
                    moves.append('P')
                else:
                    moves.append('R')
            else:
                if s[i] == 'P':
                    wins += 1
                    moves.append('S')
                elif s[i] == 'S':
                    moves.append('S')
                else:
                    moves.append('P')
                    
        
        
        
        cur_wins = wins
        cur_moves = moves[:]

        
        for i in range(n):
            if moves[i] == 'R':
                if s[i] == 'R':
                    if i > 0 and moves[i-1] != 'P':
                        
                        cur_moves[i] = 'P'
                        if s[i] == 'R':
                            cur_wins +=1
                    elif i < n-1 and moves[i+1] != 'P':
                        cur_moves[i] = 'P'
                        if s[i] == 'R':
                            cur_wins +=1
            elif moves[i] == 'P':
                if s[i] == 'P':
                    if i > 0 and moves[i-1] != 'S':
                        cur_moves[i] = 'S'
                        if s[i] == 'P':
                            cur_wins +=1
                            
                    elif i < n-1 and moves[i+1] != 'S':
                        cur_moves[i] = 'S'
                        if s[i] == 'P':
                            cur_wins +=1
            else:
                if s[i] == 'S':
                    if i > 0 and moves[i-1] != 'R':
                        cur_moves[i] = 'R'
                        if s[i] == 'S':
                            cur_wins +=1
                    elif i < n-1 and moves[i+1] != 'R':
                        cur_moves[i] = 'R'
                        if s[i] == 'S':
                            cur_wins +=1

        
        ans = max(ans, cur_wins)
        
    print(ans)

solve()