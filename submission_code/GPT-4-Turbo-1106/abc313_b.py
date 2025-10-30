def find_strongest_programmer(N, M, superiority_info):
    # Initialize a list to keep track of the number of programmers weaker than each programmer
    weaker_count = [0] * (N + 1)
    
    # Process the superiority information
    for stronger, weaker in superiority_info:
        weaker_count[weaker] += 1
    
    # Find the strongest programmer
    strongest_programmer = -1
    for i in range(1, N + 1):
        # If a programmer has no one stronger than them, they are the strongest
        if weaker_count[i] == 0:
            # If we already found a strongest programmer, it means there are multiple strongest programmers
            if strongest_programmer != -1:
                return -1
            strongest_programmer = i
    
    return strongest_programmer

# Read input from stdin
N, M = map(int, input().split())
superiority_info = [tuple(map(int, input().split())) for _ in range(M)]

# Find and print the strongest programmer
print(find_strongest_programmer(N, M, superiority_info))