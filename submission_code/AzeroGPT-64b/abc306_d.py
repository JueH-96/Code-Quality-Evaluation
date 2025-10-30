N = int(input())
X_y = [list(map(int, input().split())) for _ in range(N)]

# first course must be skipped if poison
res = X_y[0][1] if X_y[0][0] == 0 else 0

# initialize max values
before_healthy = 0
after_healthy = res
before_sick = 0
after_sick = res

# check the sum starting from each course
for i in range(1, N):
    xi, yi = X_y[i][0], X_y[i][1]
    # determine the max sum with healthy stomach (i.e. before and after eating this course)
    if xi == 0:  # health course
        before_healthy, after_healthy = max(before_healthy, before_sick), before_healthy + yi
    else:  # poison course
        before_healthy, after_healthy = max(before_healthy, after_sick), after_sick
    # determine the max sum with sick stomach 
    if xi == 0:  # health course
        before_sick, after_sick = max(before_sick, after_healthy), after_healthy
    else:  # poison course
        before_sick, after_sick = max(before_sick, after_sick), before_sick + yi
    # update res (necessary because we count starting from each course)
    res = max(res, after_healthy, after_sick, before_healthy, before_sick)

print(res)