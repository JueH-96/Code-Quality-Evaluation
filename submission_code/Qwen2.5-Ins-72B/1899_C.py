# YOUR CODE HERE
import sys

def max_alternating_subarray_sum(arr):
    n = len(arr)
    if n == 1:
        return arr[0]
    
    dp_even = [0] * n
    dp_odd = [0] * n
    
    dp_even[0] = arr[0] if arr[0] % 2 == 0 else -float('inf')
    dp_odd[0] = arr[0] if arr[0] % 2 != 0 else -float('inf')
    
    for i in range(1, n):
        if arr[i] % 2 == 0:
            dp_even[i] = max(dp_even[i-1], dp_odd[i-1] + arr[i], arr[i])
            dp_odd[i] = dp_odd[i-1]
        else:
            dp_odd[i] = max(dp_odd[i-1], dp_even[i-1] + arr[i], arr[i])
            dp_even[i] = dp_even[i-1]
    
    return max(max(dp_even), max(dp_odd))

input = sys.stdin.read
data = input().split()
index = 0
t = int(data[index])
index += 1
results = []

for _ in range(t):
    n = int(data[index])
    index += 1
    arr = list(map(int, data[index:index+n]))
    index += n
    results.append(max_alternating_subarray_sum(arr))

for result in results:
    print(result)