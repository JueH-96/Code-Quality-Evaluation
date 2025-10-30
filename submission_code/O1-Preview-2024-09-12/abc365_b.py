# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))
list_of_tuples = list(enumerate(A,1))  # (position, value)
sorted_list = sorted(list_of_tuples, key=lambda x:x[1], reverse=True)
print(sorted_list[1][0])