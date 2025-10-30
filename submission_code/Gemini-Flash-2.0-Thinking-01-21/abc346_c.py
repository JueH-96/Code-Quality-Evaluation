def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    present_numbers = set()
    for x in a:
        if 1 <= x <= k:
            present_numbers.add(x)
            
    sum_present = sum(present_numbers)
    total_sum = (k * (k + 1)) // 2
    missing_sum = total_sum - sum_present
    print(missing_sum)

if __name__ == '__main__':
    solve()