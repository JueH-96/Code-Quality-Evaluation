import sys

rating = int(sys.stdin.readline().strip())

if 1 <= rating <= 99:
    target_display = 2
elif 100 <= rating <= 199:
    target_display = 3
elif 200 <= rating <= 299:
    target_display = 4
else:
    raise ValueError("Rating must be between 1 and 299 inclusive")

current_display = 1
if 1 <= rating <= 99:
    pass
elif 100 <= rating <= 199:
    current_display = 2
elif 200 <= rating <= 299:
    current_display = 3

print(target_display - current_display)