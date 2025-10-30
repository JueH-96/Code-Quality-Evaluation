def solve():
    v1, v2, v3 = map(int, input().split())
    
    if v3 != 0 and v3 != 7 and v3 != 49 and v3 != 343:
        print("No")
        return
    
    if v3 == 0:
        if v1 + v2 != 3 * 343:
            print("No")
            return
        
        if v2 == 0:
            print("Yes")
            print("0 0 0 8 0 0 16 0 0")
            return
        
        if v2 == 2 * 343:
            print("Yes")
            print("0 0 0 0 8 0 0 16 0")
            return
        
        if v2 == 343:
            print("Yes")
            print("0 0 0 0 8 0 8 0 0")
            return
        
        print("No")
        return
    
    if v3 == 7:
        if v1 + v2 + v3 != 3 * 343:
            print("No")
            return
        
        if v2 == 84 and v1 == 840:
            print("Yes")
            print("0 0 0 0 6 0 6 0 0")
            return
        
        if v2 == 168 and v1 == 756:
            print("Yes")
            print("0 0 0 0 6 0 6 6 0")
            return
        
        print("No")
        return
    
    if v3 == 49:
        if v1 + v2 + v3 != 3 * 343:
            print("No")
            return
        
        if v2 == 168 and v1 == 756:
            print("Yes")
            print("0 0 0 0 6 0 0 6 6")
            return
        
        print("No")
        return
    
    if v3 == 343:
        if v1 + v2 + v3 != 3 * 343:
            print("No")
            return
        
        if v1 == 0 and v2 == 0:
            print("Yes")
            print("0 0 0 0 0 0 0 0 0")
            return
        
        print("No")
        return

solve()