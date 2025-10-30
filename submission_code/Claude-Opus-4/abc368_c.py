# YOUR CODE HERE
n = int(input())
healths = list(map(int, input().split()))

T = 0

for health in healths:
    # For the current enemy, we need to reduce health to 0 or less
    # Starting from current T+1
    
    while health > 0:
        T += 1
        if T % 3 == 0:
            health -= 3
        else:
            health -= 1

print(T)