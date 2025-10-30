def solve():
    n, q = map(int, input().split())
    queries = []
    for _ in range(q):
        queries.append(int(input()))
    
    possible_fixed_counts = {}
    possible_fixed_counts[2] = {0, 4}
    possible_fixed_counts[3] = {0, 9}
    possible_fixed_counts[29] = {0, 108, 321, 681, 841}
    
    results = []
    for k in queries:
        if n in possible_fixed_counts:
            if k in possible_fixed_counts[n]:
                results.append("Yes")
            else:
                results.append("No")
        else:
            if k == 0 or k == n*n:
                results.append("Yes")
            else:
                results.append("No")
                
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()