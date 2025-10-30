N, T, P = map(int, input().split())
L = list(map(int, input().split()))

# Count how many people currently have hair length >= T
current_count = sum(1 for length in L if length >= T)

# If already satisfied, return 0
if current_count >= P:
    print(0)
else:
    # Calculate days needed for each person to reach T
    days_needed = []
    for length in L:
        if length < T:
            days_needed.append(T - length)
    
    # Sort the days needed
    days_needed.sort()
    
    # We need (P - current_count) more people
    additional_needed = P - current_count
    
    # The answer is the (additional_needed)-th smallest days needed
    print(days_needed[additional_needed - 1])