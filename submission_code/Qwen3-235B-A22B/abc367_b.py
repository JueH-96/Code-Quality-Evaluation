s = input().strip()
left_part, right_part = s.split('.')
num_left = int(left_part)
trimmed_right = right_part.rstrip('0')
if not trimmed_right:
    print(str(num_left))
else:
    print(f"{num_left}.{trimmed_right}")