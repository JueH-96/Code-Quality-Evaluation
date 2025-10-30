def solve():
    n = int(input())
    s = input()

    t_wins = 0
    a_wins = 0

    for char in s:
        if char == 'T':
            t_wins += 1
        else:
            a_wins += 1

    if t_wins > a_wins:
        print('T')
    elif a_wins > t_wins:
        print('A')
    else:
        t_wins_current = 0
        a_wins_current = 0
        t_reached_final = False
        a_reached_final = False
        for char in s:
            if char == 'T':
                t_wins_current += 1
            else:
                a_wins_current += 1

            if t_wins_current == t_wins and not t_reached_final:
                t_reached_final = True
            if a_wins_current == a_wins and not a_reached_final:
                a_reached_final = True

            if t_reached_final and not a_reached_final:
                print('T')
                return
            elif a_reached_final and not t_reached_final:
                print('A')
                return
        print('T')

solve()