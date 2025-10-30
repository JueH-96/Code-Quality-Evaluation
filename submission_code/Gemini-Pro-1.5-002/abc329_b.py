# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

largest = max(a)
second_largest = -1

for x in a:
    if x != largest:
        if second_largest == -1 or x > second_largest:
            second_largest = x

print(second_largest)