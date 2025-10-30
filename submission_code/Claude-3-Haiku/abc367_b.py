import sys

x = float(input())

if x == 0:
    print(0)
else:
    print(f"{x:.3f}".rstrip("0").rstrip("."))