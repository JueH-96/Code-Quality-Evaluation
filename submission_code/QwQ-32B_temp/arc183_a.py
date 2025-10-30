def main():
    import sys
    N, K = map(int, sys.stdin.readline().split())
    counts = [K] * N
    result = []
    T_ratio = 0.5

    while sum(counts) > 0:
        m = sum(counts)
        current_cum = 0.0
        chosen_i = -1
        for i in range(N):
            if counts[i] == 0:
                continue
            frac = counts[i] / m
            if current_cum + frac >= T_ratio:
                chosen_i = i
                break
            else:
                current_cum += frac
        # Append the chosen number (1-based)
        result.append(chosen_i + 1)
        counts[chosen_i] -= 1
        # Update T_ratio
        frac = (counts[chosen_i] + 1) / m
        new_T_ratio = (T_ratio - current_cum) / frac
        T_ratio = new_T_ratio

    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()