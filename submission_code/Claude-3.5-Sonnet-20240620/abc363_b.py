# YOUR CODE HERE
N, T, P = map(int, input().split())
L = list(map(int, input().split()))

days = 0
while True:
    count = sum(1 for length in L if length >= T)
    if count >= P:
        print(days)
        break
    L = [length + 1 for length in L]
    days += 1