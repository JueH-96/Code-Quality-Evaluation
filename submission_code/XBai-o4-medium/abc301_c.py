import sys
from collections import Counter

def main():
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    count_S = Counter(S)
    count_T = Counter(T)
    
    allowed = {'a', 't', 'c', 'o', 'd', 'e', 'r'}
    
    # Check non-allowed and non-@ characters
    for c in 'abcdefghijklmnopqrstuvwxyz':
        if c not in allowed and c != '@':
            if count_S.get(c, 0) != count_T.get(c, 0):
                print("No")
                return
    
    # Calculate sum of allowed characters in S and T
    sum_s_allowed = 0
    sum_t_allowed = 0
    for c in allowed:
        sum_s_allowed += count_S.get(c, 0)
        sum_t_allowed += count_T.get(c, 0)
    
    s_at = count_S.get('@', 0)
    t_at = count_T.get('@', 0)
    
    if (sum_s_allowed + s_at) != (sum_t_allowed + t_at):
        print("No")
        return
    
    # Compute sum_min
    sum_min = 0
    for c in allowed:
        s_c = count_S.get(c, 0)
        t_c = count_T.get(c, 0)
        delta = s_c - t_c
        sum_min += max(0, -delta)
    
    if sum_min <= s_at:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()