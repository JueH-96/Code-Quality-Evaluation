def solve():
    n = int(input())
    x = list(map(int, input().split()))

    def calculate_sum(arr):
        return sum(arr)

    def operate(arr, i):
        m = (arr[i] + arr[i+3]) / 2
        new_arr = arr[:]
        new_arr[i+1] = m + (m - arr[i+1])
        new_arr[i+2] = m - (arr[i+2] - m)
        return new_arr

    import itertools
    
    min_sum = float('inf')
    
    
    def find_min_sum(arr):
        nonlocal min_sum
        current_sum = calculate_sum(arr)
        min_sum = min(min_sum, current_sum)
        
        for i in range(n - 3):
            new_arr = operate(arr, i)
            find_min_sum(new_arr)