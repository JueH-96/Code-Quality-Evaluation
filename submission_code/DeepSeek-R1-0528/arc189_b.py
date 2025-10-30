n = int(input().strip())
A = list(map(int, input().split()))

if n == 1:
    print(A[0])
else:
    d = [A[i+1] - A[i] for i in range(n-1)]
    even_d = []
    odd_d = []
    for i in range(len(d)):
        if i % 2 == 0:
            even_d.append(d[i])
        else:
            odd_d.append(d[i])
            
    even_d.sort()
    odd_d.sort()
    
    total = n * A[0]
    for j in range(len(d)):
        if j % 2 == 0:
            total += (n - 1 - j) * even_d[j // 2]
        else:
            total += (n - 1 - j) * odd_d[j // 2]
            
    print(total)