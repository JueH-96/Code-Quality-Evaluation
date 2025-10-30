# YOUR CODE HERE
def is_repdigit(n):
    return len(set(str(n))) == 1

N = int(input())
days = list(map(int, input().split()))

count = 0
for month, day_count in enumerate(days, 1):
    if is_repdigit(month):
        count += sum(1 for day in range(1, day_count + 1) if is_repdigit(day))

print(count)