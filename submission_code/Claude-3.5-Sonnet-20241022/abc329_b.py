N = int(input())
A = list(map(int, input().split()))

max_num = max(A)
second_max = max(x for x in A if x != max_num)

print(second_max)