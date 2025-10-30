s = input().strip()
left, right = s.split('.')
stripped_right = right.rstrip('0')
if stripped_right:
    print(f"{left}.{stripped_right}")
else:
    print(left)