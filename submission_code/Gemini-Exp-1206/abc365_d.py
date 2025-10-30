def solve():
    n = int(input())
    s = input()
    
    wins = 0
    prev_takahashi = ''
    
    for aoki in s:
        if aoki == 'R':
            if prev_takahashi != 'P':
                wins += 1
                prev_takahashi = 'P'
            else:
                prev_takahashi = 'R'
        elif aoki == 'P':
            if prev_takahashi != 'S':
                wins += 1
                prev_takahashi = 'S'
            else:
                prev_takahashi = 'P'
        elif aoki == 'S':
            if prev_takahashi != 'R':
                wins += 1
                prev_takahashi = 'R'
            else:
                prev_takahashi = 'S'
                
    print(wins)

solve()