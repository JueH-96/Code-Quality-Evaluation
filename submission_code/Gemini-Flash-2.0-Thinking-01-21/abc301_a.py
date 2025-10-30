def solve():
    n = int(input())
    s = input()
    t_wins_total = 0
    a_wins_total = 0
    for char in s:
        if char == 'T':
            t_wins_total += 1
        else:
            a_wins_total += 1
    
    if t_wins_total > a_wins_total:
        print('T')
    elif a_wins_total > t_wins_total:
        print('A')
    else:
        current_t_wins = 0
        current_a_wins = 0
        for char in s:
            if char == 'T':
                current_t_wins += 1
                if current_t_wins == t_wins_total:
                    print('T')
                    return
            elif char == 'A':
                current_a_wins += 1
                if current_a_wins == a_wins_total:
                    print('A')
                    return

if __name__ == '__main__':
    solve()