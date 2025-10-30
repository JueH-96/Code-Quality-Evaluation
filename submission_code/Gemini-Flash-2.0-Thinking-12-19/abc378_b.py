def solve():
    n = int(input())
    garbage_types = []
    for _ in range(n):
        q_r = list(map(int, input().split()))
        garbage_types.append({'q': q_r[0], 'r': q_r[1]})
    
    q_queries = int(input())
    queries = []
    for _ in range(q_queries):
        t_d = list(map(int, input().split()))
        queries.append({'type_index': t_d[0], 'day': t_d[1]})
    
    results = []
    for query in queries:
        type_index = query['type_index']
        day_put_out = query['day']
        garbage_info = garbage_types[type_index - 1]
        q_val = garbage_info['q']
        r_val = garbage_info['r']
        
        y = day_put_out - r_val
        k = (y + q_val - 1) // q_val
        next_collection_day = k * q_val + r_val
        results.append(next_collection_day)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()