# Reading input
import sys
input = sys.stdin.read
data = input().split()

# Extracting N and K
N = int(data[0])
K = int(data[1])

# Extracting the sequence A
A = list(map(int, data[2:]))

# Finding multiples of K and their quotients
quotients = []
for number in A:
    if number % K == 0:
        quotients.append(number // K)

# Sorting quotients (though they should already be sorted since A is sorted)
quotients.sort()

# Printing the result
print(" ".join(map(str, quotients)))