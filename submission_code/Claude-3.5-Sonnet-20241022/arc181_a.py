def is_sorted(arr):
    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

def is_identity(arr):
    return all(arr[i] == i+1 for i in range(len(arr)))

def apply_operation(arr, k):
    n = len(arr)
    result = arr.copy()
    
    if k >= 2:
        left = result[:k-1]
        left.sort()
        result[:k-1] = left
        
    if k <= n-1:
        right = result[k:]
        right.sort()
        result[k:] = right
        
    return result

def solve_case(n, p):
    if is_identity(p):
        return 0
        
    seen = {tuple(p)}
    queue = [(p, 0)]
    
    while queue:
        curr_p, ops = queue.pop(0)
        
        for k in range(1, n+1):
            next_p = apply_operation(curr_p, k)
            next_p_tuple = tuple(next_p)
            
            if next_p_tuple not in seen:
                if is_identity(next_p):
                    return ops + 1
                seen.add(next_p_tuple)
                queue.append((next_p, ops + 1))
                
    return -1

T = int(input())
for _ in range(T):
    N = int(input())
    P = list(map(int, input().split()))
    print(solve_case(N, P))