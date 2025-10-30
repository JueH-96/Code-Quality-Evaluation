A_str = input().split()
A = [int(x) for x in A_str]

result = 0
for i in range(64):
    result += A[i] * (2**i)

print(result)