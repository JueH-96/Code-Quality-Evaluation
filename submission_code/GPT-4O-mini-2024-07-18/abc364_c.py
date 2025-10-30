def min_dishes_to_eat(N, X, Y, A, B):
    # Create a list of dishes with their sweetness and saltiness
    dishes = list(zip(A, B))
    
    # Sort dishes based on the ratio of sweetness to saltiness
    # This helps in maximizing the sweetness while minimizing the saltiness
    dishes.sort(key=lambda dish: (dish[0] / dish[1], dish[0]), reverse=True)
    
    total_sweetness = 0
    total_saltiness = 0
    count = 0
    
    for sweetness, saltiness in dishes:
        total_sweetness += sweetness
        total_saltiness += saltiness
        count += 1
        
        if total_sweetness > X or total_saltiness > Y:
            break
    
    return count

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
Y = int(data[2])
A = list(map(int, data[3:3 + N]))
B = list(map(int, data[3 + N:3 + 2 * N]))

result = min_dishes_to_eat(N, X, Y, A, B)
print(result)