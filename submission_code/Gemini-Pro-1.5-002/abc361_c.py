# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    a_sorted = sorted(a)
    min_diff = float('inf')

    for i in range(k + 1):
        min_val = a_sorted[i]
        for j in range(k - i + 1):
            max_val = a_sorted[n - 1 - j]
            
            temp_a = []
            removed_count = 0
            
            for x in a:
                if removed_count < k:
                    if x < min_val or x > max_val:
                        removed_count += 1
                    else:
                        temp_a.append(x)
                else:
                    temp_a.append(x)
                    
            
            if len(temp_a) == n - k:
                
                if not temp_a:
                    min_diff = 0
                    break
                
                
                curr_min = float('inf')
                curr_max = float('-inf')
                for x in temp_a:
                    curr_min = min(curr_min, x)
                    curr_max = max(curr_max, x)
                min_diff = min(min_diff, curr_max - curr_min)
        
    print(min_diff)

solve()