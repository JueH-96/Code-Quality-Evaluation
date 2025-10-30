# YOUR CODE HERE
N, K = map(int, input().split())
medicines = []
for i in range(N):
    a, b = map(int, input().split())
    medicines.append((a, b))

left = 1
right = 10**18

while left <= right:
    mid = (left + right) // 2
    total_pills = 0
    for a, b in medicines:
        if mid <= a:
            total_pills += b
        else:
            total_pills += b
    if total_pills <= K:
        right = mid - 1
    else:
        left = mid + 1

ans = left
total_pills = 0
for a, b in medicines:
    if ans <= a:
        total_pills += b

while total_pills > K:
    ans +=1
    for i in range(N):
        a,b = medicines[i]
        if ans > a:
            total_pills -= b

print(ans)