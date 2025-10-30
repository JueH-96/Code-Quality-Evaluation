# YOUR CODE HERE
N, M = map(int, input().split())

limit = 10**9
total = 1  # N^0 = 1
current_term = 1

for i in range(1, M + 1):
    current_term *= N
    total += current_term
    
    # Check if we've exceeded the limit
    if total > limit:
        print("inf")
        exit()

print(total)