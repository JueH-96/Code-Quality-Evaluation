import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))
    total_all = N * (N + 1) // 2
    pos_dict = defaultdict(list)
    for idx, val in enumerate(A):
        pos_dict[val].append(idx + 1)  # Using 1-based index

    ans = 0
    for x in pos_dict:
        pos_list = pos_dict[x]
        sum_no = 0
        k = len(pos_list)
        # Left segment before the first occurrence
        left_len = pos_list[0] - 1
        sum_no += left_len * (left_len + 1) // 2
        # Middle segments between occurrences
        for i in range(1, k):
            gap = pos_list[i] - pos_list[i-1] - 1
            sum_no += gap * (gap + 1) // 2
        # Right segment after the last occurrence
        right_len = N - pos_list[-1]
        sum_no += right_len * (right_len + 1) // 2
        ans += (total_all - sum_no)
    print(ans)

if __name__ == '__main__':
    main()