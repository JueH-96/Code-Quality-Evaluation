def min_operations_to_empty_sequence(test_cases):
    results = []
    
    for A in test_cases:
        n = len(A)
        if n == 0:
            results.append(0)
            continue
        
        # Count the number of unique segments
        unique_segments = 1  # At least one segment exists
        for i in range(1, n):
            if A[i] != A[i - 1]:
                unique_segments += 1
        
        # The minimum operations required is the number of unique segments
        results.append(unique_segments)
    
    return results

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    T = int(data[0])
    test_cases = []
    index = 1
    
    for _ in range(T):
        N = int(data[index])
        A = list(map(int, data[index + 1].split()))
        test_cases.append(A)
        index += 2
    
    results = min_operations_to_empty_sequence(test_cases)
    
    # Print all results, one per line
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == "__main__":
    main()