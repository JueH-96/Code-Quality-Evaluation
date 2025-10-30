import bisect
import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    Q = int(data[ptr])
    ptr += 1
    sum_T = 0
    s_list = []
    s_values = []
    prefix_sums = []
    current_start = 0
    for _ in range(Q):
        query = data[ptr]
        if query == '1':
            s = sum_T
            if s_list and s_list[-1][0] == s:
                s_list[-1] = (s, s_list[-1][1] + 1)
                prefix_sums[-1] += 1
            else:
                s_list.append((s, 1))
                s_values.append(s)
                if prefix_sums:
                    prefix_sums.append(prefix_sums[-1] + 1)
                else:
                    prefix_sums.append(1)
            ptr += 1
        elif query == '2':
            T = int(data[ptr + 1])
            sum_T += T
            ptr += 2
        elif query == '3':
            H = int(data[ptr + 1])
            target_S = sum_T - H
            idx = bisect.bisect_right(s_values, target_S, current_start) - 1
            if idx < current_start:
                ans = 0
            else:
                if current_start == 0:
                    ans = prefix_sums[idx]
                else:
                    ans = prefix_sums[idx] - prefix_sums[current_start - 1]
            print(ans)
            current_start = idx + 1
            ptr += 2

if __name__ == '__main__':
    main()