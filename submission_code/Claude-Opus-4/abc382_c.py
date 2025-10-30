# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

for j in range(M):
    sushi_deliciousness = B[j]
    eaten = False
    
    for i in range(N):
        if sushi_deliciousness >= A[i]:
            print(i + 1)  # Person numbers are 1-indexed
            eaten = True
            break
    
    if not eaten:
        print(-1)