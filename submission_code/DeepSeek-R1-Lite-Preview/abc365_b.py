N = int(input())
A = list(map(int, input().split()))
sorted_A = sorted(A, reverse=True)
second_largest = sorted_A[1]
position = A.index(second_largest) + 1
print(position)