import sys
import bisect

def compute_sum_freq(pos, D):
    N_pos = len(pos)
    sorted_pos = sorted(pos)
    prefix_sum = [0]
    cum_sum_val = 0
    for num in sorted_pos:
        cum_sum_val += num
        prefix_sum.append(cum_sum_val)
    min_pos_val = min(pos)
    max_pos_val = max(pos)
    freq = [0] * (D + 1)
    for x_val in range(min_pos_val - D, max_pos_val + D + 1):
        P_num = bisect.bisect_right(sorted_pos, x_val)
        sum_left = prefix_sum[P_num]
        total_sum_pos = prefix_sum[N_pos]
        A_x = (2 * P_num - N_pos) * x_val + (total_sum_pos - 2 * sum_left)
        if 0 <= A_x <= D:
            freq[A_x] += 1
    return freq

data = sys.stdin.read().split()
data = [int(x) for x in data]
N = data[0]
D = data[1]
x_coords = data[2::2]
y_coords = data[3::2]

freq_A = compute_sum_freq(x_coords, D)
freq_B = compute_sum_freq(y_coords, D)

cum_B = [0] * (D + 1)
cum_B[0] = freq_B[0]
for i in range(1, D + 1):
    cum_B[i] = cum_B[i - 1] + freq_B[i]

answer = 0
for s in range(0, D + 1):
    answer += freq_A[s] * cum_B[D - s]

print(answer)