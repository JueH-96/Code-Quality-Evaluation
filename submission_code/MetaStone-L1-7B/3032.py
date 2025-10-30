def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    n, k = map(int, sys.stdin.readline().split())
    receiver = list(map(int, sys.stdin.readline().split()))
    memo = {}

    def compute_f(x):
        if x in memo:
            return memo[x]
        current = x
        path = []
        while current not in memo:
            memo[current] = {'tail': [], 'cycle': []}
            path.append(current)
            current = receiver[current]
        if current in memo:
            if current in path:
                index = path.index(current)
                memo[x]['tail'] = path[:index]
                memo[x]['cycle'] = path[index:]
            else:
                memo[x]['tail'] = path + memo[current]['tail']
                memo[x]['cycle'] = memo[current]['cycle']
        s_tail = sum(memo[x]['tail'])
        len_tail = len(memo[x]['tail'])
        s_cycle = sum(memo[x]['cycle'])
        len_cycle = len(memo[x]['cycle'])
        if k <= len_tail:
            total = sum(receiver[i] for i in path[:k])
            memo[x]['result'] = total
        else:
            full_cycles = (k - len_tail) // len_cycle
            remainder = (k - len_tail) % len_cycle
            total = s_tail + full_cycles * s_cycle
            if remainder > 0:
                total += sum(memo[x]['cycle'][:remainder])
            memo[x]['result'] = total
        return memo[x]['result']

    max_f = 0
    for x in range(n):
        current_max = compute_f(x)
        if current_max > max_f:
            max_f = current_max
    print(max_f)

if __name__ == "__main__":
    main()