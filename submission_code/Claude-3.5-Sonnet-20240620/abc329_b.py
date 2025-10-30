# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

max_num = max(A)
second_max = max(num for num in A if num != max_num)

print(second_max)