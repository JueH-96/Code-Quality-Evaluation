def solve():
    n = int(input())
    a = list(map(int, input().split()))
    m = int(input())
    b = list(map(int, input().split()))
    l = int(input())
    c = list(map(int, input().split()))
    q = int(input())
    x_queries = list(map(int, input().split()))
    
    possible_sums = set()
    for val_a in a:
        for val_b in b:
            for val_c in c:
                possible_sums.add(val_a + val_b + val_c)
                
    results = []
    for query_x in x_queries:
        if query_x in possible_sums:
            results.append("Yes")
        else:
            results.append("No")
            
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()