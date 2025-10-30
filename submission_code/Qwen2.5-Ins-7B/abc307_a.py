# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))
weeks = [sum(A[i*7:(i+1)*7]) for i in range(N)]
print(*weeks)