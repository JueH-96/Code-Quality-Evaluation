import sys
input = sys.stdin.read
data = input().split()

index = 0
t = int(data[index])
index += 1
results = []

for _ in range(t):
    n = int(data[index])
    index += 1
    a = list(map(int, data[index:index + n]))
    index += n

    max_sum = float('-inf')
    current_sum = 0
    prev_parity = None

    for num in a:
        current_parity = num % 2
        if current_parity != prev_parity:
            current_sum = max(num, current_sum + num)
        else:
            current_sum = num
        max_sum = max(max_sum, current_sum)
        prev_parity = current_parity

    results.append(max_sum)

for result in results:
    print(result)