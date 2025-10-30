# YOUR CODE HERE
n = int(input())
s = input()
count_a = 0
count_b = 0
count_c = 0
ans = 0
for i in range(n):
    if s[i] == 'A':
        count_a += 1
    elif s[i] == 'B':
        count_b += 1
    else:
        count_c += 1
    if count_a >= 1 and count_b >= 1 and count_c >= 1:
        ans = i + 1
        break
print(ans)