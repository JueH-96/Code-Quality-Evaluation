# YOUR CODE HERE
import sys
input = sys.stdin.read

def min_slimes_after_synthesis():
    data = input().split()
    N = int(data[0])
    slimes = []
    
    for i in range(N):
        S = int(data[2 * i + 1])
        C = int(data[2 * i + 2])
        slimes.append((S, C))
    
    slimes.sort()
    
    total_slimes = 0
    
    for size, count in slimes:
        total_slimes += count % 2
        if count >= 2:
            next_size = size * 2
            next_count = count // 2
            slimes.append((next_size, next_count))
    
    print(total_slimes)

min_slimes_after_synthesis()