# YOUR CODE HERE

N, T, P = map(int, input().split())
L = list(map(int, input().split()))

days = 0
while True:
    count = len([i for i in L if i + days >= T])
    if count >= P:
        print(days)
        break
    days += 1