n = int(input())
p = list(map(int, input().split()))

p1 = p[0]

if n == 1:
    # If there's only one person, they're already the strongest
    x = 0
else:
    # Find the maximum ability among all other people
    max_other = max(p[1:])
    # Calculate minimum x needed
    x = max(0, max_other - p1 + 1)

print(x)