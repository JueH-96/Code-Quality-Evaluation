# YOUR CODE HERE
def sushi_eating():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:N+2+M]))
    
    results = []
    
    for b in B:
        eaten = False
        for i in range(N):
            if b >= A[i]:
                results.append(i + 1)
                eaten = True
                break
        if not eaten:
            results.append(-1)
    
    for result in results:
        print(result)

sushi_eating()