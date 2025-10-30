# YOUR CODE HERE
def min_operations_to_remove_black_cells(t, test_cases):
    results = []
    for n, k, s in test_cases:
        operations = 0
        i = 0
        while i < n:
            if s[i] == 'B':
                operations += 1
                i += k
            else:
                i += 1
        results.append(operations)
    return results

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    t = int(data[0])
    index = 1
    test_cases = []
    for _ in range(t):
        n = int(data[index])
        k = int(data[index + 1])
        s = data[index + 2]
        test_cases.append((n, k, s))
        index += 3
    
    results = min_operations_to_remove_black_cells(t, test_cases)
    for result in results:
        print(result)