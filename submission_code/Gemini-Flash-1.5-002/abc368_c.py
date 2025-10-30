# YOUR CODE HERE
n = int(input())
h = list(map(int, input().split()))

t = 0
i = 0
while i < n:
    if h[i] <= 0:
        i += 1
        continue
    if (t + 1) % 3 == 0:
        d = 3
    else:
        d = 1
    turns_needed = (h[i] + d - 1) // d
    t += turns_needed
    h[i] -= turns_needed * d
    
print(t)