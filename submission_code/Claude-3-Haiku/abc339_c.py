import sys

N = int(input())
A = list(map(int, input().split()))

min_passengers = 0
current_passengers = 0

for a in A:
    current_passengers += a
    min_passengers = min(min_passengers, current_passengers)

print(max(0, min_passengers))