from collections import defaultdict

N = int(input())
sticks = [input() for _ in range(N)]

# Create a dictionary to store the count of each stick
stick_counts = defaultdict(int)

for stick in sticks:
    # Check if the stick is the same as its reverse
    if stick == stick[::-1]:
        stick_counts[stick] += 1
    else:
        stick_counts[stick] += 1
        stick_counts[stick[::-1]] += 1

# Count the number of unique sticks
print(len(stick_counts))