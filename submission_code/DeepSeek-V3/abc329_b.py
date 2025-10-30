# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
max_a = max(a)
second_max = -1
for num in a:
    if num < max_a and num > second_max:
        second_max = num
print(second_max)