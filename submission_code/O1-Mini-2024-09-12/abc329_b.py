n = int(input())
arr = list(map(int, input().split()))
max_val = max(arr)
filtered = [x for x in arr if x != max_val]
print(max(filtered))