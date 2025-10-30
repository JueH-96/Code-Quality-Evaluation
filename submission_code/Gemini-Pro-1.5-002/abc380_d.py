def solve():
    s = input()
    q = int(input())
    queries = list(map(int, input().split()))

    results = []
    for k in queries:
        n = len(s)
        while k > n:
            n *= 2
        
        while k > len(s):
            if k > n // 2:
                k -= n // 2
                
                if 'a' <= s[k-1] <= 'z':
                    results.append(s[k-1].upper())
                else:
                    results.append(s[k-1].lower())
            else:
                
                results.append(s[k-1])

            n //= 2
        else:
            results.append(s[k-1])
            

    print(*results)

solve()