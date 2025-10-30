def solve():
    n = int(input())
    courses = []
    for _ in range(n):
        x, y = map(int, input().split())
        courses.append((x, y))

    max_tastiness = 0

    for i in range(1 << n):
        tastiness = 0
        healthy = True
        alive = True
        
        eaten_courses = []

        for j in range(n):
            if (i >> j) & 1:
                eaten_courses.append(courses[j])
            else:
                eaten_courses.append(None)

        for j in range(n):
            if eaten_courses[j] is not None:
                x, y = eaten_courses[j]
                if healthy:
                    if x == 0:
                        tastiness += y
                    else:
                        tastiness += y
                        healthy = False
                else:
                    if x == 0:
                        tastiness += y
                        healthy = True
                    else:
                        alive = False
                        break
        
        if alive:
            max_tastiness = max(max_tastiness, tastiness)

    print(max_tastiness)

solve()