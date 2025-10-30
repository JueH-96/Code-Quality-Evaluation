import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    # Track the first male child for each family
    first_male = [False] * (N + 1)  # Using 1-based index for families
    
    index = 2
    results = []
    for _ in range(M):
        A_i = int(data[index])
        B_i = data[index + 1]
        index += 2
        
        if B_i == 'M':
            if not first_male[A_i]:
                results.append("Yes")
                first_male[A_i] = True
            else:
                results.append("No")
        else:
            results.append("No")
    
    # Print results
    for result in results:
        print(result)