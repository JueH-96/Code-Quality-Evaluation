def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_f = 0
    for l in range(1, n + 1):
        for r in range(l, n + 1):
            sub_array = a[l-1:r]
            
            def calculate_f(arr):
                if not arr:
                    return 0
                
                min_ops = float('inf')
                
                for i in range(1, len(set(arr)) + 1):
                    
                    for l_val in sorted(list(set(arr))):
                        
                        
                        
                        for r_val in sorted(list(set(arr))):
                            if r_val < l_val:
                                continue
                            
                            vals_to_remove = []
                            for val in sorted(list(set(arr))):
                                if l_val <= val <= r_val:
                                    vals_to_remove.append(val)
                            
                            
                            valid = True
                            for val in vals_to_remove:
                                if val not in arr:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            
                            
                            new_arr = []
                            for val in arr:
                                if l_val <= val <= r_val:
                                    pass
                                else:
                                    new_arr.append(val)
                            
                            
                            
                            min_ops = min(min_ops, 1 + calculate_f(new_arr))
                return min_ops
            
            total_f += calculate_f(sub_array)
    
    print(total_f)

solve()