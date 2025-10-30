import sys
data = sys.stdin.read().split()
N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:2+N]))
present_set = {num for num in A if 1 <= num <= K}
sum_present = sum(present_set)
total_sum = (K * (K + 1)) // 2
missing_sum = total_sum - sum_present
print(missing_sum)