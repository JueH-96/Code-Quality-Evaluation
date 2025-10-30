import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))
index += N
S_list = []
T_list = []
for _ in range(N - 1):
    S_val = int(data[index])
    index += 1
    T_val = int(data[index])
    index += 1
    S_list.append(S_val)
    T_list.append(T_val)
# Now compute the maximum currency for country N
curr_avail = A[0]
for j in range(N - 1):
    gain = T_list[j] * (curr_avail // S_list[j])
    curr_avail = A[j + 1] + gain
# curr_avail now holds the maximum amount of currency N
print(curr_avail)