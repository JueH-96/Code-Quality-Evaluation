def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    N, M, K = map(int, input_data[:3])
    idx = 3

    # We'll store each test's keys as a bitmask, along with the result.
    tests = []
    for _ in range(M):
        Ci = int(input_data[idx]); idx += 1
        keys = list(map(int, input_data[idx:idx+Ci]))
        idx += Ci
        result = input_data[idx]; idx += 1
        # Convert the set of keys into a bitmask
        mask = 0
        for k in keys:
            # key k corresponds to bit (k-1) in a 0-based bitmask
            mask |= (1 << (k - 1))
        tests.append((mask, result))

    answer = 0

    # Enumerate all possible ways (bitmasks) of which keys are real (1) or dummy (0).
    # N <= 15, so it's feasible to check all 2^N combinations.
    for comb in range(1 << N):
        valid = True
        for mask, res in tests:
            # Count how many real keys in (comb & mask)
            # Python 3.10+ has int.bit_count()
            real_count = (comb & mask).bit_count()

            if res == 'o':
                # Door opens => we need real_count >= K
                if real_count < K:
                    valid = False
                    break
            else:
                # Door does not open => real_count < K
                if real_count >= K:
                    valid = False
                    break
        if valid:
            answer += 1

    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()