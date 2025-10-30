import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        K = int(data[index + 1])
        index += 2
        
        A = list(map(int, data[index:index + N]))
        B = list(map(int, data[index + N:index + 2 * N]))
        index += 2 * N
        
        # Combine A and B into pairs and sort by A descending, then by B descending
        AB = sorted(zip(A, B), key=lambda x: (-x[0], -x[1]))
        
        # We need the first K elements with the smallest max(A_i) which is AB[0][0]
        max_A = AB[0][0]
        sum_B = sum(b for _, b in AB[:K])
        
        # Calculate the expression
        result = max_A * sum_B
        results.append(result)
    
    # Print all results
    for res in results:
        print(res)