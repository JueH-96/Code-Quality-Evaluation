# YOUR CODE HERE
def min_slimes():
    N = int(input())
    slimes = {}
    
    for _ in range(N):
        S, C = map(int, input().split())
        slimes[S] = C
    
    total_slimes = 0
    for size, count in slimes.items():
        while count > 0:
            if count % 2 == 1:
                total_slimes += 1
            count //= 2
    
    print(total_slimes)

min_slimes()