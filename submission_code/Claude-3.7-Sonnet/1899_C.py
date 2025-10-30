def max_alternating_parity_subarray_sum(arr):
    n = len(arr)
    
    max_odd = float("-inf")
    max_even = float("-inf")
    
    if arr[0] % 2 == 0:
        max_even = arr[0]
    else:
        max_odd = arr[0]
    
    max_sum = max(max_odd, max_even)
    
    for i in range(1, n):
        if arr[i] % 2 == 0:  # Even
            max_even = max(arr[i], max_odd + arr[i]) if max_odd != float("-inf") else arr[i]
        else:  # Odd
            max_odd = max(arr[i], max_even + arr[i]) if max_even != float("-inf") else arr[i]
        
        max_sum = max(max_sum, max_odd, max_even)
    
    return max_sum

def solve():
    t = int(input())
    for _ in range(t):
        n = int(input())
        arr = list(map(int, input().split()))
        print(max_alternating_parity_subarray_sum(arr))

solve()