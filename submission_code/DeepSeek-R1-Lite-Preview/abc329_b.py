N = int(input())
A = list(map(int, input().split()))

max_value = max(A)
filtered_A = [x for x in A if x != max_value]
second_max = max(filtered_A)

print(second_max)