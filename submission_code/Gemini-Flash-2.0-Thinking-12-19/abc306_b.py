A_str = input().split()
A = [int(x) for x in A_str]
result = 0
power_of_2 = 1
for i in range(len(A)):
    term = A[i] * power_of_2
    result += term
    power_of_2 *= 2
print(result)