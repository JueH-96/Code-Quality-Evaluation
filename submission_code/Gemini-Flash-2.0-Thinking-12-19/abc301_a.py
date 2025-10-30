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
        t_current_wins = 0
        a_current_wins = 0
        t_reach_index = -1
        a_reach_index = -1
        for i in range(n):
            if s[i] == 'T':
                t_current_wins += 1
            else:
                a_current_wins += 1
            if t_current_wins == t_wins and t_reach_index == -1:
                t_reach_index = i + 1
            if a_current_wins == a_wins and a_reach_index == -1:
                a_reach_index = i + 1

        if t_reach_index < a_reach_index:
            print('T')
        else:
            print('A')

if __name__ == "__main__":
    solve()