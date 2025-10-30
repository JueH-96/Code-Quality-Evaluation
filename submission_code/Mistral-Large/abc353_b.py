import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

count = 0
current_empty_seats = K

for i in range(N):
    if current_empty_seats < A[i]:
        count += 1
        current_empty_seats = K
    current_empty_seats -= A[i]

# After the loop, if there are any groups left or the attraction was started one last time
if current_empty_seats != K:
    count += 1

print(count)