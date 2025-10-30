s = input().strip()
upper_count = sum(1 for char in s if char.isupper())
lower_count = len(s) - upper_count

if upper_count > lower_count:
	print(s.upper())
else:
	print(s.lower())