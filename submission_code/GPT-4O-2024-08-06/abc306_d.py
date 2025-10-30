# YOUR CODE HERE
def max_tastiness(N, courses):
    total_tastiness = 0
    healthy = True  # Start with a healthy stomach
    
    for i in range(N):
        X, Y = courses[i]
        
        if healthy:
            # If healthy, consider eating both types of courses
            if X == 0:
                # Antidotal course, always safe to eat
                total_tastiness += Y
            else:
                # Poisonous course, eat only if tastiness is positive
                if Y > 0:
                    total_tastiness += Y
                    healthy = False  # Eating a poisonous course makes the stomach upset
        else:
            # If upset, only eat antidotal courses to become healthy again
            if X == 0:
                total_tastiness += Y
                healthy = True  # Eating an antidotal course makes the stomach healthy again
    
    return total_tastiness

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
courses = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N)]

print(max_tastiness(N, courses))