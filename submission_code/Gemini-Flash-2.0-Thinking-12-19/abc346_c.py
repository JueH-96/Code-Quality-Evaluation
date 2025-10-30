def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    total_sum = k * (k + 1) // 2
    present_numbers_set = set()
    for x in a:
        if 1 <= x <= k:
            present_numbers_set.add(x)
            
    sum_present = 0
    for p in present_numbers_set:
        sum_present += p
        
    result = total_sum - sum_present
    print(result)

if __name__ == '__main__':
    solve()