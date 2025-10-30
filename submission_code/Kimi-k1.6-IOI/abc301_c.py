import sys
from collections import Counter

def main():
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    allowed = {'a', 't', 'c', 'o', 'd', 'e', 'r'}
    s_counter = Counter(S)
    t_counter = Counter(T)
    
    # Check invalid characters
    invalid_chars = set()
    for c in s_counter:
        if c != '@' and c not in allowed:
            invalid_chars.add(c)
    for c in t_counter:
        if c != '@' and c not in allowed:
            invalid_chars.add(c)
    
    for c in invalid_chars:
        if s_counter.get(c, 0) != t_counter.get(c, 0):
            print("No")
            return
    
    # Calculate sum_pos and sum_neg for allowed characters
    allowed_chars = ['a', 't', 'c', 'o', 'd', 'e', 'r']
    sum_pos = 0
    sum_neg = 0
    for c in allowed_chars:
        s = s_counter.get(c, 0)
        t = t_counter.get(c, 0)
        delta = t - s
        sum_pos += max(delta, 0)
        sum_neg += max(-delta, 0)
    
    at_s = s_counter.get('@', 0)
    at_t = t_counter.get('@', 0)
    
    if sum_pos > at_s or sum_neg > at_t:
        print("No")
    elif (at_s - sum_pos) != (at_t - sum_neg):
        print("No")
    else:
        print("Yes")

if __name__ == "__main__":
    main()