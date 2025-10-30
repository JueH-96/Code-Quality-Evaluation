def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    T = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1

    incompat = []
    for _ in range(M):
        a = int(input[ptr]) - 1; ptr += 1
        b = int(input[ptr]) - 1; ptr += 1
        incompat.append((a, b))

    count = 0

    def backtrack(rgs, current_max, pos):
        nonlocal count
        if pos == N:
            if current_max != T:
                return
            # Check incompatible pairs
            valid = True
            for a, b in incompat:
                if rgs[a] == rgs[b]:
                    valid = False
                    break
            if valid:
                count += 1
            return

        # Generate possible next values
        for k in range(1, current_max + 2):
            new_rgs = rgs.copy()
            new_rgs.append(k)
            new_current_max = current_max
            if k > current_max:
                new_current_max = k
            backtrack(new_rgs, new_current_max, pos + 1)

    backtrack([1], 1, 1)
    print(count)

if __name__ == "__main__":
    main()