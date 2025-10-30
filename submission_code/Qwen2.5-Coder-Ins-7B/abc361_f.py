# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])

def count_powers(N):
    count = 0
    for a in range(2, int(N**0.5) + 1):
        b = 2
        while True:
            power = a ** b
            if power > N:
                break
            count += 1
            b += 1
    return count

result = count_powers(N)
print(result)