# YOUR CODE HERE
n, a = map(int, input().split())
times = list(map(int, input().split()))
line = [0]
for i in range(n):
    if times[i] >= line[-1]:
        line.append(times[i] + a)
    else:
        line.append(line[-1] + a)
for i in range(n):
    print(line[i+1] - times[i])