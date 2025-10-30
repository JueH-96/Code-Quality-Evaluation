# YOUR CODE HERE
N = int(input())
courses = []
for i in range(N):
    x, y = map(int, input().split())
    courses.append((x, y))

max_tastiness = 0
for i in range(1 << N):
    tastiness = 0
    stomach = 0
    alive = True
    eaten_courses = []
    for j in range(N):
        if (i >> j) & 1:
            x, y = courses[j]
            eaten_courses.append((x,y))

    
    stomach_state = 0
    current_tastiness = 0
    for x,y in eaten_courses:
        if stomach_state == 0:
            if x == 0:
                stomach_state = 0
                current_tastiness += y
            else:
                stomach_state = 1
                current_tastiness += y
        else:
            if x == 0:
                stomach_state = 0
                current_tastiness += y
            else:
                alive = False
                break
    if alive:
        max_tastiness = max(max_tastiness, current_tastiness)

print(max_tastiness)