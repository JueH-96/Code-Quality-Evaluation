N = int(input())
A = []
for i in range(N):
    row = list(map(int, input().split()))
    A.append(row)

current_element = 1
for i in range(1, N + 1):
    if current_element >= i:
        current_element = A[i - 1][current_element - 1]
    else:
        current_element = A[current_element - 1][i - 1]

print(current_element)