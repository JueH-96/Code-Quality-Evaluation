def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    queries = []
    for _ in range(q):
        l, r, L, R = map(int, input().split())
        queries.append(((l, r), (L, R)))
    
    results = []
    for (l_range, r_range) in queries:
        l_start, l_end = l_range
        r_start, r_end = r_range
        sub_a = a[l_start-1:l_end]
        sub_b = b[r_start-1:r_end]
        
        if len(sub_a) != len(sub_b):
            results.append("No")
        else:
            freq_a = {}
            for x in sub_a:
                freq_a[x] = freq_a.get(x, 0) + 1
            freq_b = {}
            for x in sub_b:
                freq_b[x] = freq_b.get(x, 0) + 1
                
            possible = True
            for num in freq_a:
                if freq_a[num] != freq_b.get(num, 0):
                    possible = False
                    break
            if possible:
                results.append("Yes")
            else:
                results.append("No")
                
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()