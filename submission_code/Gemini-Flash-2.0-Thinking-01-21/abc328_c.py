def solve():
    n, q = map(int, input().split())
    s = input()
    queries = []
    for _ in range(q):
        l, r = map(int, input().split())
        queries.append({'l': l, 'r': r})
    
    results = []
    for query in queries:
        l_i = query['l']
        r_i = query['r']
        count = 0
        for p in range(l_i - 1, r_i - 1):
            if s[p] == s[p+1]:
                count += 1
        results.append(count)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()