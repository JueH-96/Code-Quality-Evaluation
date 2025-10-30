n = int(input().strip())
arr = list(map(int, input().split()))
distinct = sorted(set(arr), reverse=True)
print(distinct[1])