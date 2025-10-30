import sys
from functools import lru_cache

def main():
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline())
    len_S = len(S)
    B = bin(N)[2:]  # Binary representation without '0b' prefix
    len_B = len(B)

    # Case 1: S is shorter than B's length
    if len_S < len_B:
        max_val = int(S.replace('?', '1'), 2)
        print(max_val)
        return

    # Case 2: S is longer or equal to B's length
    # Pad B with leading zeros to match len_S
    B_padded = '0' * (len_S - len_B) + B
    B_padded = list(B_padded)

    # Compute minimal possible value
    min_val = int(S.replace('?', '0'), 2)
    if min_val > N:
        print(-1)
        return

    # Compute maximal possible value
    max_val = int(S.replace('?', '1'), 2)
    if max_val <= N:
        print(max_val)
        return

    # Now, we need to find the maximum number <= N using the recursive approach
    @lru_cache(maxsize=None)
    def dfs(pos, is_tight):
        if pos == len_S:
            return 0
        current_char = S[pos]
        if current_char != '?':
            current_bit = int(current_char)
            if is_tight:
                if current_bit > int(B_padded[pos]):
                    return -1
                new_tight = is_tight and (current_bit == int(B_padded[pos]))
                next_val = dfs(pos + 1, new_tight)
                if next_val == -1:
                    return -1
                return (current_bit << (len_S - pos - 1)) + next_val
            else:
                next_val = dfs(pos + 1, False)
                if next_val == -1:
                    return -1
                return (current_bit << (len_S - pos - 1)) + next_val
        else:
            max_val = -1
            # Try setting to 1
            if is_tight:
                if int(B_padded[pos]) >= 1:
                    new_tight = (1 == int(B_padded[pos]))
                    next_val = dfs(pos + 1, new_tight)
                    if next_val != -1:
                        candidate = (1 << (len_S - pos - 1)) + next_val
                        if candidate <= N:
                            if candidate > max_val:
                                max_val = candidate
            else:
                next_val = dfs(pos + 1, False)
                if next_val != -1:
                    candidate = (1 << (len_S - pos - 1)) + next_val
                    if candidate <= N and candidate > max_val:
                        max_val = candidate
            # Try setting to 0
            if is_tight:
                if 0 <= int(B_padded[pos]):
                    new_tight = (0 == int(B_padded[pos]))
                    next_val = dfs(pos + 1, new_tight)
                    if next_val != -1:
                        candidate = (0 << (len_S - pos - 1)) + next_val
                        if candidate <= N and candidate > max_val:
                            max_val = candidate
            else:
                next_val = dfs(pos + 1, False)
                if next_val != -1:
                    candidate = (0 << (len_S - pos - 1)) + next_val
                    if candidate <= N and candidate > max_val:
                        max_val = candidate
            return max_val if max_val != -1 else -1

    result = dfs(0, True)
    print(result if result != -1 else -1)

if __name__ == "__main__":
    main()