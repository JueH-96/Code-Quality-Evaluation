import sys

# Read all input from stdin
data = sys.stdin.read().split()
index = 0
n = int(data[index])
index += 1
a = list(map(int, data[index:index + n]))

# Determine the winner
if n % 2 == 0:
    print("Snuke")
else:
    # Check if all elements are even
    all_even = all(x % 2 == 0 for x in a)
    if all_even and n > 1:
        print("Snuke")
    else:
        print("Fennec")