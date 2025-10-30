# Read input
N, M = map(int, input().split())
X = list(map(int, input().split()))
A = list(map(int, input().split()))

# Check if total stones is N
if sum(A) != N:
    print(-1)
    exit()

# Sort X and A based on X values
sorted_pairs = sorted(zip(X, A))
X = [x for x, a in sorted_pairs]
A = [a for x, a in sorted_pairs]

# Check if the first position is 1
if X[0] != 1:
    print(-1)
    exit()

prev_x = 0
sum_prev = 0

# Check all intervals for sum >= j
for i in range(M):
    current_x = X[i]
    sum_current = sum_prev + A[i]
    
    # Check current_x condition
    if sum_current < current_x:
        print(-1)
        exit()
    
    # Check interval between previous_x and current_x
    start = prev_x + 1
    end = current_x - 1
    if start <= end:
        if sum_prev < end:
            print(-1)
            exit()
    
    sum_prev = sum_current
    prev_x = current_x

# Check the last interval
start = prev_x + 1
end = N
if start <= end:
    if sum_prev < end:
        print(-1)
        exit()

# Calculate the minimum operations
current_position = 1
total_operations = 0
for i in range(M):
    x = X[i]
    a = A[i]
    start = max(current_position, x)
    end = start + a - 1
    
    # Check if stones fit into the available positions
    if end > N:
        print(-1)
        exit()
    
    # Compute operations for current stones
    ops = a * (start - x) + (a * (a - 1)) // 2
    total_operations += ops
    current_position = end + 1

print(total_operations)