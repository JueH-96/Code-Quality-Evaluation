# YOUR CODE HERE
n = int(input())
days = list(map(int, input().split()))
count = 0
for i, d in enumerate(days, start=1):
    for j in range(1, d+1):
        if str(i) == str(j) or str(i) == str(j)[1] or str(i) == str(j)[0]:
            count += 1
print(count)