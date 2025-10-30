import sys

numbers = list(map(int, sys.stdin.read().split()))
for num in reversed(numbers):
    print(num)