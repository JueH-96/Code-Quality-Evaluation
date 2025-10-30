from itertools import permutations

def max_dishes_eaten(N, X, Y, dishes):
    max_eaten = 0
    for perm in permutations(dishes):
        sweetness, saltiness, eaten = 0, 0, 0
        for dish in perm:
            if sweetness + dish[0] <= X and saltiness + dish[1] <= Y:
                sweetness += dish[0]
                saltiness += dish[1]
                eaten += 1
            else:
                break
        max_eaten = max(max_eaten, eaten)
    return max_eaten

# Read input from stdin
N, X, Y = map(int, input().split())
dishes = [tuple(map(int, input().split())) for _ in range(N)]

# Solve the problem
answer = max_dishes_eaten(N, X, Y, dishes)

# Write the answer to stdout
print(answer)