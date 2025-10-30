import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
S = int(data[index])
index += 1
arr = list(map(int, data[index:index + N]))
P = sum(arr)
cum_sum_list = [0]
for num in arr:
    cum_sum_list.append(cum_sum_list[-1] + num)
cum_values = cum_sum_list[:N]
cum_set = set(cum_values)
D1 = S // P
C1 = S % P
has_pair1 = any(val - C1 in cum_set for val in cum_values)
has_pair2 = False
if S % P > 0:
    C2 = (S % P) - P
    has_pair2 = any(val - C2 in cum_set for val in cum_values)
if has_pair1 or has_pair2:
    print("Yes")
else:
    print("No")