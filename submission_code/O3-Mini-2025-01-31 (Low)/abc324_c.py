def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return

    # Parse first line: N and received string T'
    parts = data[0].split()
    N = int(parts[0])
    T_prime = parts[1]
    lenT = len(T_prime)

    # The next N lines contain the candidate strings S_i.
    S_list = data[1:]
    result_indices = []
    
    for idx, S in enumerate(S_list, start=1):
        lenS = len(S)
        # Case 1: T' is exactly equal to T.
        if T_prime == S:
            result_indices.append(idx)
            continue
        
        # Case 2: T' is obtained by inserting one letter in T.
        #   Then len(T_prime) == len(T) + 1, so here S (which would be T) has lenS + 1 == lenT.
        if lenT == lenS + 1:
            i, j = 0, 0
            diff = 0
            while i < lenT and j < lenS:
                if T_prime[i] == S[j]:
                    i += 1
                    j += 1
                else:
                    diff += 1
                    i += 1  # Skip the inserted character in T_prime.
                    if diff > 1:
                        break
            # Even if j reaches the end, a leftover character in T_prime counts as one operation.
            if diff <= 1:
                result_indices.append(idx)
            continue

        # Case 3: T' is obtained by deleting one letter from T.
        #   Then len(T_prime) == len(T) - 1, so here candidate S (which would be T) is one character longer.
        if lenT == lenS - 1:
            i, j = 0, 0
            diff = 0
            while i < lenT and j < lenS:
                if T_prime[i] == S[j]:
                    i += 1
                    j += 1
                else:
                    diff += 1
                    j += 1  # Skip one character in S (the deleted one).
                    if diff > 1:
                        break
            if diff <= 1:
                result_indices.append(idx)
            continue

        # Case 4: T' is obtained by replacing one letter in T.
        #   Then len(T_prime) == len(T) and they differ in exactly one position.
        if lenT == lenS:
            diff = 0
            for a, b in zip(T_prime, S):
                if a != b:
                    diff += 1
                    if diff > 1:
                        break
            if diff == 1:
                result_indices.append(idx)
            continue

    # Output result.
    out_lines = []
    out_lines.append(str(len(result_indices)))
    if result_indices:
        out_lines.append(" ".join(map(str, result_indices)))
    sys.stdout.write("
".join(out_lines))


if __name__ == '__main__':
    main()