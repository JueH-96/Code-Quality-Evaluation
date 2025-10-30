# YOUR CODE HERE
import sys
sys.setrecursionlimit(1000000)
input = sys.stdin.read
data = input().split()
index = 0
N = int(data[index])
index += 1

num = N
while True:
    A = num // 100
    B = (num // 10) % 10
    C = num % 10
    if A * B == C:
        print(num)
        break
    num += 1