N = int(input())
A = list(map(int, input().split()))
sorted_A = sorted(A)
print(A.index(sorted_A[-2]) + 1)