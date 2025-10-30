# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        M = int(data[index + 1])
        index += 2
        
        capacities = []
        
        for _ in range(N):
            V_i = int(data[index])
            P_i = int(data[index + 1])
            index += 2
            capacities.append(V_i)
        
        # Sort capacities
        capacities.sort()
        
        # Calculate the maximum possible earnings
        max_earnings = sum(capacities)
        
        results.append(max_earnings)
    
    for result in results:
        print(result)