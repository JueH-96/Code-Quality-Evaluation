def min_operations_to_sort_permutation(test_cases):
    results = []
    
    for N, P in test_cases:
        # Count the number of contiguous segments that are out of order
        segments = 0
        for i in range(1, N):
            if P[i] < P[i - 1]:
                segments += 1
        
        # If there are no segments, it's already sorted
        if segments == 0:
            results.append(0)
        else:
            # The number of operations needed is segments + 1
            results.append(segments + 1)
    
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
        P = list(map(int, data[index + 1].split()))
        test_cases.append((N, P))
        index += 2
    
    results = min_operations_to_sort_permutation(test_cases)
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()