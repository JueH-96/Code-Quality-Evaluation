def main():
    import sys
    import math

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD = 998244353

    # d[g] = (count, total_score) meaning:
    #  - 'count' = number of subsequences that end with gcd=g
    #  - 'total_score' = sum of scores of those subsequences
    d = {}
    results = []
    for i in range(N):
        x = A[i]
        # Start a new dictionary for the next state
        new_d = {x: (1, 0)}  # single-element subsequence from x has count=1, score=0
        for g, (count, total_score) in d.items():
            new_g = math.gcd(g, x)
            # Each existing subsequence adds new_g to its score if extended with x
            # So the new total_score for those extended subsequences is old total_score + new_g * count
            ext_count = count
            ext_score = (total_score + new_g * count) % MOD

            if new_g in new_d:
                old_count, old_score = new_d[new_g]
                new_d[new_g] = ((old_count + ext_count) % MOD, (old_score + ext_score) % MOD)
            else:
                new_d[new_g] = (ext_count, ext_score)

        # new_d becomes the dictionary for the next iteration
        d = new_d

        # The answer for m = i+1 is sum of total_score for all gcd states
        # (because single-element subsequences contribute 0 to pairs, so they're already accounted for)
        ans = 0
        for _, (cnt, tscore) in d.items():
            ans = (ans + tscore) % MOD
        results.append(ans)

    print('
'.join(map(str, results)))


if __name__ == "__main__":
    main()