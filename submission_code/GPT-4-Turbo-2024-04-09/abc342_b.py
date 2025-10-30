import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    
    P = list(map(int, data[index:index+N]))
    index += N
    
    Q = int(data[index])
    index += 1
    
    results = []
    
    for _ in range(Q):
        A = int(data[index]) - 1
        B = int(data[index+1]) - 1
        index += 2
        
        # Since A < B and we need the person closer to the front (smaller index)
        # We need to find the person with the smaller index in the list P
        if P[A] < P[B]:
            results.append(P[A])
        else:
            results.append(P[B])
    
    for result in results:
        print(result)