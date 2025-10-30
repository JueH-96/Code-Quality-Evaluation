def solve():
    N = int(input())
    A = input().strip()
    
    def build_tree(start, end, level):
        if level == 0:
            return {
                'value': int(A[start]),
                'is_leaf': True
            }
        else:
            length = end - start
            child_length = length // 3
            
            children = []
            for i in range(3):
                child_start = start + i * child_length
                child_end = start + (i + 1) * child_length
                children.append(build_tree(child_start, child_end, level - 1))
            
            values = [child['value'] for child in children]
            majority = 1 if sum(values) >= 2 else 0
            
            return {
                'value': majority,
                'is_leaf': False,
                'children': children
            }
    
    root = build_tree(0, len(A), N)
    
    def compute_flip_cost(node):
        if node['is_leaf']:
            return 1
        
        children = node['children']
        flip_costs = [compute_flip_cost(child) for child in children]
        
        current_values = [child['value'] for child in children]
        target_majority = 1 - node['value']
        
        min_cost = float('inf')
        
        # Try all 8 possible combinations of flipping/keeping each child
        for flip_mask in range(8):
            total_cost = 0
            final_values = []
            
            for i in range(3):
                if flip_mask & (1 << i):
                    total_cost += flip_costs[i]
                    final_values.append(1 - current_values[i])
                else:
                    final_values.append(current_values[i])
            
            final_majority = 1 if sum(final_values) >= 2 else 0
            
            if final_majority == target_majority:
                min_cost = min(min_cost, total_cost)
        
        return min_cost
    
    print(compute_flip_cost(root))

solve()