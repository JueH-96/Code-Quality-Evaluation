def main():
    import sys
    N, K = map(int, sys.stdin.readline().split())
    from math import factorial
    from sys import stdin

    counts = [K] * (N + 1)  # 1-based indexing
    result = []
    remaining = N * K
    k = (factorial(N * K) // factorial(K) ** N + 1) // 2
    k -= 1  # convert to 0-based index

    for _ in range(N * K):
        # Sort the numbers by current count (descending), then by value (ascending)
        sorted_nums = sorted(range(1, N+1), key=lambda x: (-counts[x], x))
        total = 0
        chosen = None
        for num in sorted_nums:
            cnt = counts[num]
            if cnt == 0:
                continue
            # Compute how many sequences start with this number
            # Which is (remaining-1)! / ( (counts[num]-1)! * product (counts[i]!) )
            # But since this is hard to compute, use the ratio
            # We need to find the first num where the cumulative count >= k
            # Since count is proportional to cnt, use the cumulative cnt
            # However, this is an approximation
            if total + cnt > k:
                chosen = num
                break
            total += cnt
        if chosen is None:
            chosen = sorted_nums[0]
        result.append(chosen)
        counts[chosen] -= 1
        remaining -= 1

    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()