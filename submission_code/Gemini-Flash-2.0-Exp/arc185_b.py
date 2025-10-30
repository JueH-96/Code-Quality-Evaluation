def solve():
    n = int(input())
    a = list(map(int, input().split()))

    total_sum = sum(a)
    expected_sum = n * (n - 1) // 2
    
    if total_sum < expected_sum:
        print("No")
        return
    
    diff = total_sum - expected_sum
    
    b = [i for i in range(n)]