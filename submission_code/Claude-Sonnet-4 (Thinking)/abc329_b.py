n = int(input())
a = list(map(int, input().split()))

max_val = max(a)
not_max = [x for x in a if x != max_val]
answer = max(not_max)
print(answer)