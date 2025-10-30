# YOUR CODE HERE
N = int(input())
count = 0
for i in range(N):
    s = input().strip()
    if s == "Takahashi":
        count += 1
print(count)