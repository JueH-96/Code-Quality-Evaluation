def solve():
    # Read input
    N = int(input().strip())
    integers = list(map(int, input().strip().split()))
    
    # Count occurrences of each integer
    integer_count = {}
    for A in integers:
        integer_count[A] = integer_count.get(A, 0) + 1
    
    # Find the person with the largest unique integer
    max_value = -1
    max_label = -1
    
    for i in range(N):
        if integer_count[integers[i]] == 1 and integers[i] > max_value:
            max_value = integers[i]
            max_label = i + 1  # Convert 0-indexed to 1-indexed
    
    return max_label

print(solve())