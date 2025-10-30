def can_eat_k_dishes(N, X, Y, dishes, k):
    # Try all possible combinations of k dishes
    from itertools import permutations
    
    # Get all possible k-length combinations of dishes
    for perm in permutations(range(N), k):
        sweet_sum = 0
        salt_sum = 0
        valid = True
        
        # Check if this permutation is valid
        for i in perm:
            sweet_sum += dishes[i][0]
            salt_sum += dishes[i][1]
            if sweet_sum > X or salt_sum > Y:
                valid = False
                break
        
        if valid:
            return True
            
    return False

# Read input
N, X, Y = map(int, input().split())
dishes = []
for _ in range(N):
    a, b = map(int, input().split())
    dishes.append((a, b))

# Binary search on the answer
left = 0
right = N
answer = 0

while left <= right:
    mid = (left + right) // 2
    if can_eat_k_dishes(N, X, Y, dishes, mid):
        answer = mid
        left = mid + 1
    else:
        right = mid - 1

print(answer)