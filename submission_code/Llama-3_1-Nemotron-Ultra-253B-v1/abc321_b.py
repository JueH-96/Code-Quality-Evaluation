n, x = map(int, input().split())
a_list = list(map(int, input().split()))

for a in range(101):
    temp = a_list + [a]
    sorted_temp = sorted(temp)
    sum_score = sum(sorted_temp[1:-1])
    if sum_score >= x:
        print(a)
        exit()

print(-1)