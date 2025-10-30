s = list(map(int, input().split()))

# Check condition 1: monotonically non-decreasing
cond1 = True
for i in range(7):
    if s[i] > s[i+1]:
        cond1 = False
        break

# Check condition 2: all between 100 and 675 inclusive
cond2 = True
for num in s:
    if not (100 <= num <= 675):
        cond2 = False
        break

# Check condition 3: all multiples of 25
cond3 = True
for num in s:
    if num % 25 != 0:
        cond3 = False
        break

if cond1 and cond2 and cond3:
    print("Yes")
else:
    print("No")