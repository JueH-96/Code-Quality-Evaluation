import sys
input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1
results = []

for _ in range(T):
    N = int(data[index])
    index += 1
    count = 0
    
    for _ in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        C = int(data[index + 2])
        index += 3
        
        if A == 0 and B == 0:
            if C > 0:
                count += 1
        elif A == 0:
            if B < C:
                count += 1
        elif B == 0:
            if A < C:
                count += 1
        else:
            if A * B < C:
                count += 1
    
    results.append(count)

for result in results:
    print(result)