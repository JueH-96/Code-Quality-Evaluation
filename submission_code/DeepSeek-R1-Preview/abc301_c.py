from collections import defaultdict

def main():
    S = input().strip()
    T = input().strip()
    
    allowed = {'a', 't', 'c', 'o', 'd', 'e', 'r'}
    
    count_S = defaultdict(int)
    count_T = defaultdict(int)
    at_S = 0
    at_T = 0
    
    for c in S:
        if c == '@':
            at_S += 1
        else:
            count_S[c] += 1
    
    for c in T:
        if c == '@':
            at_T += 1
        else:
            count_T[c] += 1
    
    # Check for non-allowed and non-@ characters
    all_chars = set(count_S.keys()).union(set(count_T.keys()))
    for c in all_chars:
        if c not in allowed and c != '@':
            if count_S.get(c, 0) != count_T.get(c, 0):
                print("No")
                return
    
    sum_L = 0
    for c in allowed:
        d = count_T.get(c, 0) - count_S.get(c, 0)
        sum_L += max(0, -d)
    
    if sum_L > at_T:
        print("No")
    else:
        print("Yes")
    
if __name__ == "__main__":
    main()