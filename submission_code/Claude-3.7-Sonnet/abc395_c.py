def find_shortest_subarray_with_repeated_value(arr, n):
    min_length = float('inf')
    
    for i in range(n):
        seen = set()
        for j in range(i, n):
            if arr[j] in seen:
                min_length = min(min_length, j - i + 1)
                if min_length == 2:  # Smallest possible value, return immediately
                    return 2
                break
            seen.add(arr[j])
    
    return min_length if min_length != float('inf') else -1

n = int(input())
arr = list(map(int, input().split()))
print(find_shortest_subarray_with_repeated_value(arr, n))