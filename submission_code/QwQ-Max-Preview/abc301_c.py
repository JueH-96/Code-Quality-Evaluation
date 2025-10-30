from collections import Counter

def main():
    allowed = {'a', 't', 'c', 'o', 'd', 'e', 'r'}
    s = input().strip()
    t = input().strip()
    
    s_counter = Counter(s)
    t_counter = Counter(t)
    
    # Check for characters not in allowed and not '@' with differing counts
    # Check all characters in S
    for c in s_counter:
        if c != '@' and c not in allowed:
            if s_counter[c] != t_counter.get(c, 0):
                print("No")
                return
    # Check all characters in T that are not in S
    for c in t_counter:
        if c != '@' and c not in allowed:
            if s_counter.get(c, 0) != t_counter[c]:
                print("No")
                return
    
    # Calculate sum_min
    sum_min = 0
    s_at = s_counter.get('@', 0)
    t_at = t_counter.get('@', 0)
    
    for c in allowed:
        s_count = s_counter.get(c, 0)
        t_count = t_counter.get(c, 0)
        delta = s_count - t_count
        sum_min += max(delta, 0)
    
    if sum_min <= t_at:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()