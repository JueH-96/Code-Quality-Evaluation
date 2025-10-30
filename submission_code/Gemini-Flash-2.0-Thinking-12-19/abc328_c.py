def solve():
    n, q = map(int, input().split())
    s = input()
    queries = []
    for _ in range(q):
        l, r = map(int, input().split())
        queries.append({'l': l, 'r': r})
    
    doubles_array = []
    for i in range(n - 1):
        if s[i] == s[i+1]:
            doubles_array.append(1)
        else:
            doubles_array.append(0)
            
    prefix_sum = [0] * n
    for i in range(1, n):
        prefix_sum[i] = prefix_sum[i-1] + doubles_array[i-1]
        
    results = []
    for query in queries:
        l_i = query['l']
        r_i = query['r']
        if r_i <= l_i:
            results.append(0)
        else:
            start_index = l_i - 1
            end_index = r_i - 1
            count = 0
            if end_index - start_index >= 1:
                count = prefix_sum[end_index] - prefix_sum[start_index]
            else:
                count = 0
            results.append(count)
            
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()