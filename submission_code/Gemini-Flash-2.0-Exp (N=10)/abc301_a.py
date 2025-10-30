def solve():
    n = int(input())
    s = input()
    
    t_wins = 0
    a_wins = 0
    
    t_first_win_count = -1
    a_first_win_count = -1
    
    for i in range(n):
        if s[i] == 'T':
            t_wins += 1
            if t_first_win_count == -1 and t_wins == (n + 1) // 2:
                t_first_win_count = i + 1
        else:
            a_wins += 1
            if a_first_win_count == -1 and a_wins == (n + 1) // 2:
                a_first_win_count = i + 1
                
    if t_wins > a_wins:
        print("T")
    elif a_wins > t_wins:
        print("A")
    else:
        if t_first_win_count < a_first_win_count:
            print("T")
        else:
            print("A")

solve()