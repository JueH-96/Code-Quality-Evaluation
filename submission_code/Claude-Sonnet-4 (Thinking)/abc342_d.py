def get_reduced_form(n):
    if n == 0:
        return 0
    
    reduced = 1
    i = 2
    while i * i <= n:
        count = 0
        while n % i == 0:
            n //= i
            count += 1
        if count % 2 == 1:
            reduced *= i
        i += 1
    
    if n > 1:
        reduced *= n
    
    return reduced

N = int(input())
A = list(map(int, input().split()))

zeros = 0
groups = {}

for a in A:
    if a == 0:
        zeros += 1
    else:
        rf = get_reduced_form(a)
        groups[rf] = groups.get(rf, 0) + 1

answer = 0

# Pairs involving at least one zero
answer += zeros * (N - zeros)  # One zero, one non-zero
answer += zeros * (zeros - 1) // 2  # Both zeros

# Pairs involving only non-zero numbers
for count in groups.values():
    answer += count * (count - 1) // 2

print(answer)