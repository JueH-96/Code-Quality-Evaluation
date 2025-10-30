# YOUR CODE HERE
import sys

def max_tastiness(N, courses):
    healthy = 0
    upset = -float('inf')
    
    for X, Y in courses:
        if X == 0:  # Antidotal course
            new_healthy = max(healthy, upset + Y)
            new_upset = upset
        else:  # Poisonous course
            new_healthy = healthy
            new_upset = max(upset, healthy + Y)
        
        healthy, upset = new_healthy, new_upset
    
    return max(healthy, upset)

N = int(input())
courses = [tuple(map(int, input().split())) for _ in range(N)]

print(max_tastiness(N, courses))