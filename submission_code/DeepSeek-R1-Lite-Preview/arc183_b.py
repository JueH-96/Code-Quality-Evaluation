import sys
from bisect import bisect_left, bisect_right

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        K = int(input[ptr + 1])
        ptr += 2
        A = list(map(int, input[ptr:ptr + N]))
        ptr += N
        B = list(map(int, input[ptr:ptr + N]))
        ptr += N
        # Build a mapping from value to sorted list of positions (0-based)
        value_positions = {}
        for idx, val in enumerate(A):
            if val not in value_positions:
                value_positions[val] = []
            value_positions[val].append(idx)
        possible = True
        for i in range(N):
            b = B[i]
            if b not in value_positions:
                possible = False
                break
            pos_list = value_positions[b]
            # Find if there's any position j in [i-K, i+K]
            left = i - K
            if left < 0:
                left = 0
            right = i + K
            if right >= N:
                right = N - 1
            # Find the first position >= left
            # Find the first position > right
            l = bisect_left(pos_list, left)
            r = bisect_right(pos_list, right)
            if l < r:
                continue
            else:
                possible = False
                break
        print("Yes" if possible else "No")

if __name__ == '__main__':
    main()