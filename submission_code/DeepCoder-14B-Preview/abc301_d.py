def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline())

    n = len(S)

    # Precompute min_rem and max_rem for each position
    min_rem = [0] * (n + 1)
    max_rem = [0] * (n + 1)

    for i in range(n-1, -1, -1):
        c = S[i]
        if c == '0':
            min_rem[i] = min_rem[i+1] * 2
            max_rem[i] = max_rem[i+1] * 2
        elif c == '1':
            min_rem[i] = min_rem[i+1] * 2 + 1
            max_rem[i] = max_rem[i+1] * 2 + 1
        else:
            min_rem[i] = min_rem[i+1] * 2
            max_rem[i] = (max_rem[i+1] * 2) + 1

    def backtrack(pos, current_val):
        if pos == n:
            return current_val if current_val <= N else -1
        max_val = -1
        if S[pos] == '0':
            new_val = current_val * 2
            min_rem_current = min_rem[pos+1]
            if new_val + min_rem_current > N:
                return -1
            res = backtrack(pos + 1, new_val)
            if res != -1:
                max_val = res
        elif S[pos] == '1':
            new_val = current_val * 2 + 1
            min_rem_current = min_rem[pos+1]
            if new_val + min_rem_current > N:
                return -1
            res = backtrack(pos + 1, new_val)
            if res != -1:
                max_val = res
        else:
            # Try setting to 1
            new_val_1 = current_val * 2 + 1
            min_rem_current = min_rem[pos+1]
            if new_val_1 + min_rem_current <= N:
                res_1 = backtrack(pos + 1, new_val_1)
                if res_1 != -1:
                    if res_1 > max_val:
                        max_val = res_1
            # Try setting to 0
            new_val_0 = current_val * 2
            min_rem_current = min_rem[pos+1]
            if new_val_0 + min_rem_current <= N:
                res_0 = backtrack(pos + 1, new_val_0)
                if res_0 != -1:
                    if res_0 > max_val:
                        max_val = res_0
        return max_val if max_val != -1 else -1

    result = backtrack(0, 0)
    print(result if result != -1 else -1)

if __name__ == '__main__':
    main()