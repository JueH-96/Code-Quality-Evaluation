import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    a = [int(data[i]) for i in range(2, 2*N, 2)]
    b = [int(data[i]) for i in range(3, 2*N+1, 2)]
    
    total_pills = [0] * (max(a) + 1)
    
    for i in range(N):
        for j in range(a[i] + 1):
            total_pills[j] += b[i]
    
    for day, pills in enumerate(total_pills):
        if pills <= K:
            print(day + 1)
            return

solve()