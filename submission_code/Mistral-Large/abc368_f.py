import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Count the number of times each integer appears in the list
count = {}
for num in A:
    if num in count:
        count[num] += 1
    else:
        count[num] = 1

# Determine the winner
winner = "Bruno"
for num, freq in count.items():
    if freq % 2 == 1:
        winner = "Anna"
        break

print(winner)