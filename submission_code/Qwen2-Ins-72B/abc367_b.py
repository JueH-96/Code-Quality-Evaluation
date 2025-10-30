x = float(input())
print(f'{x:f}'.rstrip('0').rstrip('.') if x != 0 else '0')