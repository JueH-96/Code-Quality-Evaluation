def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])
    K = int(data[1])
    P = int(data[2])
    
    plans = []
    index = 3
    for _ in range(N):
        C = int(data[index])
        A = list(map(int, data[index+1:index+1+K]))
        plans.append((C, A))
        index += 1 + K
    
    min_cost = float('inf')
    found = False
    
    # Iterate over all subsets of plans using bitmasking
    for mask in range(1 << N):
        total_cost = 0
        parameters = [0] * K
        
        for i in range(N):
            if mask & (1 << i):
                total_cost += plans[i][0]
                for j in range(K):
                    parameters[j] += plans[i][1][j]
        
        if all(param >= P for param in parameters):
            found = True
            min_cost = min(min_cost, total_cost)
    
    if found:
        print(min_cost)
    else:
        print(-1)