# YOUR CODE HERE
def count_equal_combinations(n, a):
    count = {}
    result = 0
    
    for i in range(n):
        if a[i] in count:
            result += count[a[i]]
        
        for j in range(i):
            if a[i] == a[j] or (a[i] % a[j] == 0 and a[i] // a[j] == a[j] // a[i]):
                result += 1
        
        count[a[i]] = count.get(a[i], 0) + 1
    
    return result

t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    print(count_equal_combinations(n, a))