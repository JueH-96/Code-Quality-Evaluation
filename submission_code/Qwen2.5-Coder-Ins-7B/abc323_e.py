# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
T = list(map(int, data[2:]))

MOD = 998244353

# Calculate the total length of all songs
total_length = sum(T)

# Calculate the number of songs that are longer than X + 0.5 seconds
count = sum(1 for t in T if t > X + 0.5)

# The probability is the number of favorable outcomes divided by the total number of outcomes
# Since each song is chosen with equal probability, the probability is simply the number of songs longer than X + 0.5
probability = count

# Print the probability modulo 998244353
print(probability)