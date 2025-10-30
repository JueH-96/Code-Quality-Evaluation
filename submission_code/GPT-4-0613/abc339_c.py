N = int(input().strip())
A = list(map(int, input().strip().split()))

passengers = 0
min_passengers = 0

for i in range(N):
    passengers += A[i]
    if passengers < min_passengers:
        min_passengers = passengers

print(abs(min_passengers))