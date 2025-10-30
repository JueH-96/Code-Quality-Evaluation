n = int(input())
x = list(map(int, input().split()))
initial_sum = sum(x)
total_reduction = 0

for i in range(len(x) - 3):
    a = x[i]
    b = x[i+1]
    c = x[i+2]
    d = x[i+3]
    if b + c > a + d:
        total_reduction += (b + c - a - d)

minimal_sum = initial_sum - 2 * total_reduction
print(minimal_sum)