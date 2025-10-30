def solve():
    s, t = input().split()
    n = len(s)
    len_t = len(t)
    if len_t > n:
        print("No")
        return
    
    for w in range(1, n):
        for c in range(1, w + 1):
            constructed_t = ""
            start_index = 0
            while start_index < n:
                end_index = min(start_index + w, n)
                substring = s[start_index:end_index]
                if len(substring) >= c:
                    constructed_t += substring[c-1]
                start_index += w
            if constructed_t == t:
                print("Yes")
                return
                
    print("No")

if __name__ == '__main__':
    solve()