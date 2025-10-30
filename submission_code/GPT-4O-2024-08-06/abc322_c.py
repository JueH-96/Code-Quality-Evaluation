# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    
    results = [0] * N
    
    # Iterate over the fireworks days
    for j in range(M):
        current_firework_day = A[j]
        if j == 0:
            start_day = 1
        else:
            start_day = A[j-1] + 1
        
        for i in range(start_day, current_firework_day):
            results[i-1] = current_firework_day - i
    
    for result in results:
        print(result)

main()