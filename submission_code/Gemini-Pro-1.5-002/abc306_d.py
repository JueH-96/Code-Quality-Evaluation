# YOUR CODE HERE
def solve():
    n = int(input())
    courses = []
    for _ in range(n):
        courses.append(list(map(int, input().split())))

    def get_tastiness(choices):
        tastiness = 0
        health = 0  # 0: healthy, 1: upset
        for i in range(n):
            if choices[i] == 1:
                if health == 0:
                    if courses[i][0] == 0:
                        tastiness += courses[i][1]
                    else:
                        tastiness += courses[i][1]
                        health = 1
                else:
                    if courses[i][0] == 0:
                        tastiness += courses[i][1]
                        health = 0
                    else:
                        return -float('inf')
        return tastiness

    max_tastiness = 0
    for i in range(2**n):
        choices = []
        for j in range(n):
            if (i >> j) & 1:
                choices.append(1)
            else:
                choices.append(0)
        
        tastiness = get_tastiness(choices)
        max_tastiness = max(max_tastiness, tastiness)

    print(max_tastiness)

solve()