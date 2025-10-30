import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    results = []
    for deliciousness in B:
        eater = -1
        for i in range(N):
            if deliciousness >= A[i]:
                eater = i + 1
                break
        results.append(eater)
    
    for result in results:
        print(result)