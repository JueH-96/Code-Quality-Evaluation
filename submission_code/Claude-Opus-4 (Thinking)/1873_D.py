t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input()
    
    operations = 0
    i = 0
    
    while i < n:
        if s[i] == 'B':
            operations += 1
            i += k
        else:
            i += 1
    
    print(operations)