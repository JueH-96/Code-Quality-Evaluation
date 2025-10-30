# YOUR CODE HERE

N = int(input())
A = list(map(int, input().split()))
Q = int(input())

sleep_times = [0] * (A[-1] + 1)
for i in range(0, N, 2):
    sleep_times[A[i]] += 1
    sleep_times[A[i+1]] -= 1

for i in range(1, len(sleep_times)):
    sleep_times[i] += sleep_times[i-1]

for i in range(1, len(sleep_times)):
    sleep_times[i] += sleep_times[i-1]

for _ in range(Q):
    l, r = map(int, input().split())
    print(sleep_times[r] - sleep_times[l-1] if l > 0 else sleep_times[r])