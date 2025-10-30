# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))
cnt = 0

while sum(1 for x in A if x > 0) > 1:
    A.sort(reverse=True)
    A[0] -= 1
    A[1] -= 1
    cnt += 1

print(cnt)