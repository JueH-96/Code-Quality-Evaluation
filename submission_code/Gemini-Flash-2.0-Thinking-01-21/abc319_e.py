def solve():
    n, x, y = map(int, input().split())
    pt_pairs = []
    for _ in range(n - 1):
        pt_pairs.append(list(map(int, input().split())))
    q = int(input())
    queries = []
    for _ in range(q):
        queries.append(int(input()))
    
    results = []
    for start_time in queries:
        current_time = start_time
        current_time += x # Walk to bus stop 1
        for i in range(n - 1):
            p_i, t_i = pt_pairs[i]
            wait_time = (p_i - (current_time % p_i)) % p_i
            current_time += wait_time # Wait for bus at stop i
            current_time += t_i # Travel to bus stop i+1
        current_time += y # Walk to Aoki's house
        results.append(current_time)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()