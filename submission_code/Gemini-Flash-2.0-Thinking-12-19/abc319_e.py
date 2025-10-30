def solve():
    n, x, y = map(int, input().split())
    p_t_pairs = []
    for _ in range(n - 1):
        p_t_pairs.append(list(map(int, input().split())))
    q = int(input())
    queries = []
    for _ in range(q):
        queries.append(int(input()))
    
    results = []
    for start_time in queries:
        current_time = start_time
        current_time += x # walk to bus stop 1
        for i in range(n - 1):
            p_i, t_i = p_t_pairs[i]
            departure_time = ((current_time + p_i - 1) // p_i) * p_i
            current_time = departure_time + t_i # arrive at next bus stop
        current_time += y # walk to Aoki's house
        results.append(current_time)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()