import sys

def main():
    N, T, M = map(int, sys.stdin.readline().split())
    pairs = []
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        pairs.append((a-1, b-1))  # Convert to 0-based indices

    def generate_rgs(n, t):
        result = []
        def backtrack(pos, current_max, current):
            if pos == n:
                if current_max == t:
                    result.append(current.copy())
                return
            for val in range(1, current_max + 2):
                new_max = max(current_max, val)
                new_current = current + [val]
                backtrack(pos + 1, new_max, new_current)
        # Start with the first element (position 0) already set to 1
        backtrack(1, 1, [1])
        return result

    rgs_list = generate_rgs(N, T)
    count = 0
    for rgs in rgs_list:
        valid = True
        for a, b in pairs:
            if rgs[a] == rgs[b]:
                valid = False
                break
        if valid:
            count += 1
    print(count)

if __name__ == "__main__":
    main()