# YOUR CODE HERE
n = int(input())
arr = list(map(int, input().split()))
max_val = max(arr)
second_max = -1
for a in arr:
    if a < max_val and (second_max == -1 or a > second_max):
        second_max = a
print(second_max)