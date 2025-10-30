n = int(input())
s = input().strip()

if n % 2 != 0:
    mid = (n + 1) // 2  # 1-based middle position
    slash_index = mid - 1  # 0-based index of '/'
    if s[slash_index] == '/':
        # Check first part (all '1's)
        first_valid = all(c == '1' for c in s[:slash_index])
        # Check third part (all '2's)
        third_valid = all(c == '2' for c in s[slash_index + 1:])
        if first_valid and third_valid:
            print("Yes")
            exit()
print("No")