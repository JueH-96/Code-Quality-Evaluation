N, K = map(int, input().split())
A = set(map(int, input().split()))

total = 0
for i in range(1, K+1):
    if i not in A:
        total += i
        
print(total)