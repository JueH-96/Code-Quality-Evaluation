# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
max_val = max(a)
a.remove(max_val)
second_max_val = max(a)
print(a.index(second_max_val) + 1)