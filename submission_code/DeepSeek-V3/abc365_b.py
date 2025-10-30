# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
indexed_a = [(value, idx + 1) for idx, value in enumerate(a)]
sorted_a = sorted(indexed_a, key=lambda x: x[0], reverse=True)
second_largest_index = sorted_a[1][1]
print(second_largest_index)