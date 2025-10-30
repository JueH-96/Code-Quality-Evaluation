a, b, c = map(int, input().split())

total_sum = a + b + c

if total_sum % 2 == 0 and max(a, b, c) <= total_sum // 2:
    if a == b == c:
        print("Yes")
    elif a + b == c or a + c == b or b + c == a:
        print("Yes")
    elif total_sum % 3 == 0 and a == total_sum // 3 and b == total_sum // 3 and c == total_sum // 3:
        print("Yes")

    elif total_sum % 2 == 0:
        if a + b == total_sum // 2 or a + c == total_sum // 2 or b + c == total_sum // 2:
            print("Yes")

        elif a == total_sum // 2 or b == total_sum // 2 or c == total_sum // 2:
            print("Yes")
        else:
            print("No")
    else:
        print("No")

elif total_sum % 3 == 0 and a == b and b == c:
    print("Yes")
elif total_sum % 3 == 0 and a == total_sum // 3 and b == total_sum // 3 and c == total_sum // 3:
    print("Yes")


else:
    print("No")