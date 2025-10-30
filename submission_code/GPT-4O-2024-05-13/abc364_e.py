# YOUR CODE HERE
def max_dishes(N, X, Y, dishes):
    from itertools import permutations
    
    max_count = 0
    
    for perm in permutations(dishes):
        total_sweetness = 0
        total_saltiness = 0
        count = 0
        
        for sweetness, saltiness in perm:
            if total_sweetness + sweetness <= X and total_saltiness + saltiness <= Y:
                total_sweetness += sweetness
                total_saltiness += saltiness
                count += 1
            else:
                break
        
        max_count = max(max_count, count)
    
    return max_count

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
Y = int(data[2])
dishes = [(int(data[2*i+3]), int(data[2*i+4])) for i in range(N)]

print(max_dishes(N, X, Y, dishes))