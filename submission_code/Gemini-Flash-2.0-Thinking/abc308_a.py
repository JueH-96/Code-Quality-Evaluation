s_str = input().split()
s = [int(x) for x in s_str]

is_non_decreasing = True
for i in range(1, 8):
    if s[i] < s[i-1]:
        is_non_decreasing = False
        break

is_in_range = True
for num in s:
    if not (100 <= num <= 675):
        is_in_range = False
        break

is_multiple_of_25 = True
for num in s:
    if num % 25 != 0:
        is_multiple_of_25 = False
        break

if is_non_decreasing and is_in_range and is_multiple_of_25:
    print("Yes")
else:
    print("No")