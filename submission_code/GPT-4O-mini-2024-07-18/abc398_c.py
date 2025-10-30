def find_unique_person_with_max_value():
    import sys
    from collections import Counter

    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Count occurrences of each integer
    count = Counter(A)
    
    # Initialize variables to track the maximum unique value and its index
    max_value = -1
    person_label = -1
    
    for i in range(N):
        if count[A[i]] == 1:  # Check if the integer is unique
            if A[i] > max_value:  # Check if it's the largest unique integer
                max_value = A[i]
                person_label = i + 1  # Store the label (1-indexed)
    
    print(person_label if person_label != -1 else -1)