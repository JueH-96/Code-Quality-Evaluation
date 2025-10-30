# YOUR CODE HERE
import sys

def find_first_day(N, K, medicines):
    pill_count = [0] * (10**9 + 1)
    
    for a, b in medicines:
        pill_count[1] += b
        pill_count[a + 1] -= b
    
    for day in range(1, 10**9 + 1):
        pill_count[day] += pill_count[day - 1]
        if pill_count[day] <= K:
            return day

input = sys.stdin.read
data = input().split()
N = int(data[0])
K = int(data[1])
medicines = [(int(data[2 * i + 2]), int(data[2 * i + 3])) for i in range(N)]

print(find_first_day(N, K, medicines))