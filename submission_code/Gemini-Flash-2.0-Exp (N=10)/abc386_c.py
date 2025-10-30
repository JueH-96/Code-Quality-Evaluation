def solve():
    k = int(input())
    s = input()
    t = input()
    
    if s == t:
        print("Yes")
        return

    n = len(s)
    m = len(t)
    
    if abs(n - m) > 1:
        print("No")
        return
    
    if abs(n-m) == 1:
        if n > m:
            
            i = 0
            j = 0
            diff_count = 0
            while i < n and j < m:
                if s[i] == t[j]:
                    i+=1
                    j+=1
                else:
                    diff_count += 1
                    i+=1
            if diff_count <= 1:
                print("Yes")
                return
            else:
                print("No")
                return
        else:
            i = 0
            j = 0
            diff_count = 0
            while i < n and j < m:
                if s[i] == t[j]:
                    i+=1
                    j+=1
                else:
                    diff_count += 1
                    j+=1
            if diff_count <= 1:
                print("Yes")
                return
            else:
                print("No")
                return
    else:
        diff_count = 0
        for i in range(n):
            if s[i] != t[i]:
                diff_count += 1
        if diff_count <= 1:
            print("Yes")
        else:
            print("No")
        
solve()