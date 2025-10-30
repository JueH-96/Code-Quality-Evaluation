def solve():
    n, k = map(int, input().split())
    medicines = []
    max_a = 0
    for _ in range(n):
        a_i, b_i = map(int, input().split())
        medicines.append({'a': a_i, 'b': b_i})
        max_a = max(max_a, a_i)
    
    low = 1
    high = max_a + 1
    result_day = -1
    
    while low <= high:
        mid_day = (low + high) // 2
        total_pills_on_mid_day = 0
        for medicine in medicines:
            if medicine['a'] >= mid_day:
                total_pills_on_mid_day += medicine['b']
                
        if total_pills_on_mid_day <= k:
            result_day = mid_day
            high = mid_day - 1
        else:
            low = mid_day + 1
            
    print(result_day)

if __name__ == '__main__':
    solve()