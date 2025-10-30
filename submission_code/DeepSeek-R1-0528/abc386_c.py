def main():
    k_val = int(input().strip())
    S = input().strip()
    T = input().strip()
    
    if S == T:
        print("Yes")
        return
        
    if k_val == 0:
        print("No")
        return
        
    n = len(S)
    m = len(T)
    
    if n == m:
        diff_count = 0
        for i in range(n):
            if S[i] != T[i]:
                diff_count += 1
                if diff_count > 1:
                    break
        print("Yes" if diff_count == 1 else "No")
    elif n == m + 1:
        i, j = 0, 0
        skipped = False
        while j < m:
            if i < n and S[i] == T[j]:
                i += 1
                j += 1
            else:
                if not skipped:
                    skipped = True
                    i += 1
                else:
                    break
        print("Yes" if j == m else "No")
    elif n == m - 1:
        i, j = 0, 0
        skipped = False
        while i < n:
            if j < m and S[i] == T[j]:
                i += 1
                j += 1
            else:
                if not skipped and j < m:
                    skipped = True
                    j += 1
                else:
                    break
        if i == n:
            if (skipped and j == m) or (not skipped and j == m - 1):
                print("Yes")
            else:
                print("No")
        else:
            print("No")
    else:
        print("No")

if __name__ == "__main__":
    main()