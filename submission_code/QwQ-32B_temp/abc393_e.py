import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N

    if N == 0:
        return

    max_A = max(A) if A else 0

    # Precompute divisors for each number up to max_A
    divisors = [[] for _ in range(max_A + 1)]
    for d in range(1, max_A + 1):
        for multiple in range(d, max_A + 1, d):
            divisors[multiple].append(d)

    # Sort each divisors list in descending order
    for x in range(1, max_A + 1):
        divisors[x].sort(reverse=True)

    # Compute frequency array
    freq = [0] * (max_A + 1)
    for num in A:
        freq[num] += 1

    # Compute cnt array
    cnt = [0] * (max_A + 1)
    for x in range(1, max_A + 1):
        if freq[x] > 0:
            for d in divisors[x]:
                cnt[d] += freq[x]

    # Process each element in A
    for num in A:
        for G in divisors[num]:
            if cnt[G] >= K:
                print(G)
                break

if __name__ == "__main__":
    main()