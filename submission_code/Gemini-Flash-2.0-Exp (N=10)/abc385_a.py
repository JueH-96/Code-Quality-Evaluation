a, b, c = map(int, input().split())
total_sum = a + b + c

if total_sum % 2 != 0:
    print("No")
else:
    half_sum = total_sum // 2
    if a == half_sum or b == half_sum or c == half_sum:
        print("Yes")
    elif a + b == half_sum or a + c == half_sum or b + c == half_sum:
        print("Yes")
    elif a + b + c == half_sum * 2 and (a != 0 or b != 0 or c != 0):
        print("Yes")
    else:
        print("No")