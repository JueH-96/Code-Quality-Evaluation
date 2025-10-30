def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    # Count wins
    count_T = S.count('T')
    count_A = S.count('A')
    
    if count_T > count_A:
        print('T')
    elif count_A > count_T:
        print('A')
    else:
        # Find the game number when each reached count_T wins
        t_wins = 0
        a_wins = 0
        t_last_win = -1
        a_last_win = -1
        for i in range(N):
            if S[i] == 'T':
                t_wins += 1
                if t_wins == count_T:
                    t_last_win = i + 1
            else:
                a_wins += 1
                if a_wins == count_A:
                    a_last_win = i + 1
        # Compare the game numbers
        if t_last_win < a_last_win:
            print('T')
        else:
            print('A')

if __name__ == "__main__":
    main()