def min_operations_to_match(T, test_cases):
    results = []
    
    for case in test_cases:
        N, A, B = case
        pieces = []
        target = []
        
        # Collect positions of pieces in A and target positions in B
        for i in range(N):
            if A[i] == '1':
                pieces.append(i)
            if B[i] == '1':
                target.append(i)
        
        # If the number of pieces and targets do not match, it's impossible
        if len(pieces) != len(target):
            results.append(-1)
            continue
        
        # Calculate the minimum number of operations needed
        operations = 0
        for p, t in zip(pieces, target):
            operations += abs(p - t)
        
        results.append(operations)
    
    return results

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    T = int(data[0])
    test_cases = []
    
    index = 1
    for _ in range(T):
        N = int(data[index])
        A = data[index + 1]
        B = data[index + 2]
        test_cases.append((N, A, B))
        index += 3
    
    results = min_operations_to_match(T, test_cases)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()