def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    max_sum = float('-inf')
    
    for i in range(n):
        for j in range(i, n):
            sub_array = a[i:j+1]
            if not sub_array:
                continue
            
            valid = True
            for k in range(len(sub_array) - 1):
                if sub_array[k] % 2 == sub_array[k+1] % 2:
                    valid = False
                    break
            
            if valid:
                max_sum = max(max_sum, sum(sub_array))
    
    print(max_sum)

t = int(input())
for _ in range(t):
    solve()