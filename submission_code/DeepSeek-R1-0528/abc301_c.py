def main():
    S = input().strip()
    T = input().strip()
    
    from collections import Counter
    cntS = Counter(S)
    cntT = Counter(T)
    
    U = "atcoder"
    U_set = set(U)
    
    all_chars = set(S) | set(T)
    for c in all_chars:
        if c not in U_set and c != '@':
            if cntS.get(c, 0) != cntT.get(c, 0):
                print("No")
                return
                
    A = 0
    B = 0
    for c in U_set:
        A += cntS.get(c, 0)
        B += cntT.get(c, 0)
        
    total_at_S = cntS.get('@', 0)
    total_at_T = cntT.get('@', 0)
    
    if total_at_S - total_at_T != B - A:
        print("No")
        return
        
    deficit_S = 0
    for c in U_set:
        cnt_s = cntS.get(c, 0)
        cnt_t = cntT.get(c, 0)
        if cnt_t > cnt_s:
            deficit_S += (cnt_t - cnt_s)
            
    if total_at_S < deficit_S:
        print("No")
        return
        
    print("Yes")

if __name__ == "__main__":
    main()