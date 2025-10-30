import sys

def find_winner(N, S):
    t_count = a_count = 0
    t_first_win = a_first_win = None
    
    for i, char in enumerate(S):
        if char == 'T':
            t_count += 1
            if t_first_win is None:
                t_first_win = i
        else:
            a_count += 1
            if a_first_win is None:
                a_first_win = i
    
    if t_count > a_count:
        return 'T'
    elif a_count > t_count:
        return 'A'
    else:
        return 'T' if t_first_win < a_first_win else 'A'

if __name__ == "__main__":
    N = int(input().strip())
    S = input().strip()
    print(find_winner(N, S))