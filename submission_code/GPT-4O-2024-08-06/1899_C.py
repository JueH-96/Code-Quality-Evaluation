# YOUR CODE HERE
def max_alternating_subarray_sum(test_cases):
    results = []
    for n, array in test_cases:
        max_sum = -float('inf')
        current_sum = array[0]
        
        for i in range(1, n):
            if (array[i] % 2) != (array[i-1] % 2):
                current_sum += array[i]
            else:
                max_sum = max(max_sum, current_sum)
                current_sum = array[i]
        
        max_sum = max(max_sum, current_sum)
        results.append(max_sum)
    
    return results

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    t = int(data[index])
    index += 1
    test_cases = []
    
    for _ in range(t):
        n = int(data[index])
        index += 1
        array = list(map(int, data[index:index + n]))
        index += n
        test_cases.append((n, array))
    
    results = max_alternating_subarray_sum(test_cases)
    for result in results:
        print(result)