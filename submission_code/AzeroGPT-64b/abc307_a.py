N = int(input())
A = list(map(int, input().split()))

B = [sum(A[i:i+7]) for i in range(0, 7*N, 7)]
print(*B)