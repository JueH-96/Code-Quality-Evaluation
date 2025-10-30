def solve():
    s_str = input().split()
    s = [int(x) for x in s_str]

    if len(s) != 8:
        raise ValueError("Input should have 8 integers")

    # Condition 1: Monotonically non-decreasing
    is_non_decreasing = True
    for i in range(7):
        if s[i] > s[i+1]:
            is_non_decreasing = False
            break

    # Condition 2: Between 100 and 675 inclusive
    is_in_range = True
    for val in s:
        if not (100 <= val <= 675):
            is_in_range = False
            break

    # Condition 3: Multiples of 25
    is_multiple_of_25 = True
    for val in s:
        if val % 25 != 0:
            is_multiple_of_25 = False
            break

    if is_non_decreasing and is_in_range and is_multiple_of_25:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()