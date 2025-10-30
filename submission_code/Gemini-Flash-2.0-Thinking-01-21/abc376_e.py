def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    indexed_pairs = []
    for i in range(n):
        indexed_pairs.append({'a': a[i], 'b': b[i], 'index': i+1})
    
    indexed_pairs.sort(key=lambda x: x['a'])
    
    min_expression_value = float('inf')
    
    for j_index in range(k - 1, n):
        current_pairs = indexed_pairs[:j_index+1]
        current_pairs.sort(key=lambda x: x['b'])
        
        subset_indices = []
        subset_b_sum = 0
        subset_a_values = []
        
        for i in range(k):
            subset_indices.append(current_pairs[i]['index'])
            subset_b_sum += current_pairs[i]['b']
            subset_a_values.append(current_pairs[i]['a'])
            
        max_a_in_subset = 0
        for index in subset_indices:
            max_a_in_subset = max(max_a_in_subset, a[index-1])
            
        current_expression_value = max_a_in_subset * subset_b_sum
        min_expression_value = min(min_expression_value, current_expression_value)
        
    print(min_expression_value)

t = int(input())
for _ in range(t):
    solve()