def max_dishes(N, X, Y, dishes):
    from itertools import permutations

    max_count = 0

    for perm in permutations(dishes):
        total_sweetness = 0
        total_saltiness = 0
        count = 0
        
        for sweetness, saltiness in perm:
            total_sweetness += sweetness
            total_saltiness += saltiness
            
            if total_sweetness > X or total_saltiness > Y:
                break
            
            count += 1
        
        max_count = max(max_count, count)

    return max_count

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N, X, Y = map(int, data[0].split())
dishes = [tuple(map(int, line.split())) for line in data[1:N+1]]

# Get the result
result = max_dishes(N, X, Y, dishes)

# Print the result
print(result)