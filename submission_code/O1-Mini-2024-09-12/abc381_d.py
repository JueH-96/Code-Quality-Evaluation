# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    def max_unique_pairs(pair_list):
        left = 0
        max_len = 0
        last_seen = {}
        for right, p in enumerate(pair_list):
            if p is None:
                left = right + 1
                last_seen.clear()
            else:
                if p in last_seen:
                    left = max(left, last_seen[p] + 1)
                last_seen[p] = right
                max_len = max(max_len, right - left + 1)
        return max_len * 2

    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    res = 0
    for parity in [0,1]:
        pair_list = []
        i = parity
        while i+1 < N:
            if A[i] == A[i+1]:
                pair_list.append(A[i])
            else:
                pair_list.append(None)
            i += 2
        res = max(res, max_unique_pairs(pair_list))
    print(res)

if __name__ == "__main__":
    main()