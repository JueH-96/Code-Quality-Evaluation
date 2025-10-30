from collections import deque, defaultdict

def main():
    n = int(input())
    a = [int(x) for x in input().split()]
    s = input().strip()

    pattern = 'MEX'
    ans = 0
    pattern_starts = defaultdict(deque)
    for i in range(n - 1, -1, -1):
        pattern_starts[s[i]].append(i)

        if len(pattern_starts) != 3:
            continue

        if ''.join(sorted(s[x] for x in pattern_starts.keys())).strip() != pattern:
            pattern_starts[s[i]].popleft()
            continue

        min_previous_index = min(pattern_starts[s[j]][-2] for j in range(3) if s[j] != s[i])
        max_previous_index = max(pattern_starts[s[j]][0] for j in range(3) if s[j] != s[i])

        res = 0
        if all(s[i] != a[i] for i in [max_previous_index, min_previous_index, i]):
            res = 0
        else:
            mex_options = [0, 1, 2]
            for one of a[max_previous_index], a[min_previous_index], a[i]:
                mex_options.remove(one)
            res = mex_options[0]
        ans += res * sum(c[0] < min_previous_index and c[-1] > max_previous_index for c in pattern_starts.values())
        
        pattern_starts[s[i]].popleft()

    print(ans)

main()