# YOUR CODE HERE
n, k = map(int, input().split())
a = list(map(int, input().split()))

if k >= n - 1:
    print(a[-1] - a[0])
else:
    min_vals = [float('inf')] * (n - k)
    max_vals = [float('-inf')] * (n - k)
    
    for i in range(n - k):
        min_vals[i] = a[i]
        max_vals[i] = a[i]
        
        for j in range(i + 1, i + k + 1):
            min_vals[i] = min(min_vals[i], a[j])
            max_vals[i] = max(max_vals[i], a[j])
    
    min_diff = float('inf')
    for i in range(n - k):
        min_diff = min(min_diff, max_vals[i] - min_vals[i + 1])
    
    print(min_diff)