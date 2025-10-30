x = float(input())
if x == int(x):
    print(int(x))
else:
    print(f'{x:.10f}'.rstrip('0').rstrip('.'))