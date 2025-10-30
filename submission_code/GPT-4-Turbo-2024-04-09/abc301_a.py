def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    t_wins = 0
    a_wins = 0
    t_reach = -1
    a_reach = -1
    
    for i in range(N):
        if S[i] == 'T':
            t_wins += 1
        elif S[i] == 'A':
            a_wins += 1
        
        # Check if one reaches the majority first
        if t_reach == -1 and t_wins > N // 2:
            t_reach = i
        if a_reach == -1 and a_wins > N // 2:
            a_reach = i
    
    if t_wins > a_wins:
        print('T')
    elif a_wins > t_wins:
        print('A')
    else:
        # If they are tied, the one who reached the tie first wins
        if t_reach != -1 and (a_reach == -1 or t_reach < a_reach):
            print('T')
        else:
            print('A')

if __name__ == "__main__":
    main()