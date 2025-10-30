def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    unique_values = sorted(list(set(a)))
    u = len(unique_values)
    min_diff = float('inf')
    
    if u == 0:
        print(0)
        return
        
    for i in range(u):
        for j in range(i, u):
            lower_bound = unique_values[i]
            upper_bound = unique_values[j]
            count_in_range = 0
            for x in a:
                if lower_bound <= x <= upper_bound:
                    count_in_range += 1
            if count_in_range >= (n - k):
                current_diff = upper_bound - lower_bound
                min_diff = min(min_diff, current_diff)
                
    print(min_diff)

if __name__ == '__main__':
    solve()