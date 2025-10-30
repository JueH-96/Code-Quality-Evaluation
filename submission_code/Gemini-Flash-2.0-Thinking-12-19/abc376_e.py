def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    indexed_a = []
    for i in range(n):
        indexed_a.append({'value': a[i], 'b_value': b[i], 'index': i+1})
    
    indexed_a.sort(key=lambda x: x['value'])
    
    min_expression_value = float('inf')
    
    for i in range(k - 1, n):
        current_max_a = indexed_a[i]['value']
        current_index = indexed_a[i]['index']
        current_b = indexed_a[i]['b_value']
        
        possible_indices = []
        for j in range(i):
            possible_indices.append({'b_value': indexed_a[j]['b_value'], 'index': indexed_a[j]['index']})
            
        if len(possible_indices) < k - 1:
            continue
            
        possible_indices.sort(key=lambda x: x['b_value'])
        
        sum_b_rest = 0
        selected_indices = [current_index]
        sum_b_rest += current_b
        
        indices_for_sum = []
        for j in range(k - 1):
            indices_for_sum.append(possible_indices[j]['index'])
            sum_b_rest += possible_indices[j]['b_value']
            
        expression_value = current_max_a * sum_b_rest
        min_expression_value = min(min_expression_value, expression_value)
        
    print(min_expression_value)

t = int(input())
for _ in range(t):
    solve()