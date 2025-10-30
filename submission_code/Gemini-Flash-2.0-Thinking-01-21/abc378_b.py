def solve():
    n = int(input())
    garbage_types = []
    for _ in range(n):
        q_r = list(map(int, input().split()))
        garbage_types.append({'q': q_r[0], 'r': q_r[1]})
    
    q_count = int(input())
    queries = []
    for _ in range(q_count):
        t_d = list(map(int, input().split()))
        queries.append({'type_index': t_d[0], 'day': t_d[1]})
    
    results = []
    for query in queries:
        type_index = query['type_index']
        day = query['day']
        params = garbage_types[type_index - 1]
        q_val = params['q']
        r_val = params['r']
        
        remainder = day % q_val
        if remainder == r_val:
            results.append(day)
        elif remainder < r_val:
            results.append(day + (r_val - remainder))
        else:
            results.append(day + (q_val - remainder + r_val))
            
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()