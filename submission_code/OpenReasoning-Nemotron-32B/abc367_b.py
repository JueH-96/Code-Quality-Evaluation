s = input().strip()
if '.' in s:
	integer_part, fractional_part = s.split('.')
	fractional_part = fractional_part.rstrip('0')
	if fractional_part:
		print(f"{integer_part}.{fractional_part}")
	else:
		print(integer_part)
else:
	print(s)