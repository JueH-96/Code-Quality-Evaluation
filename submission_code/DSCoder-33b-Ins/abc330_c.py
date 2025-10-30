# YOUR CODE HERE

D = int(input())

def min_value(D):
    x = 0
    while x*x <= D:
        y = 0
        while x*x + y*y <= D:
            if x*x + y*y == D:
                return 0
            y += 1
        x += 1
    return min(D - (x-1)**2 - (y-1)**2, (x-1)**2 + (y-1)**2 - D)

print(min_value(D))