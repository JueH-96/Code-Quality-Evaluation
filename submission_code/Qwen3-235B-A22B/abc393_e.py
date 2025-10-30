import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    if N == 0:
        return

    max_A = max(A)

    # Precompute divisors for each number up to max_A
    divisors = [[] for _ in range(max_A + 1)]
    for d in range(1, max_A + 1):
        for multiple in range(d, max_A + 1, d):
            divisors[multiple].append(d)

    # Compute frequency array
    freq = [0] * (max_A + 2)
    for x in A:
        freq[x] += 1

    # Compute count array
    count = [0] * (max_A + 2)
    for d in range(1, max_A + 1):
        for multiple in range(d, max_A + 1, d):
            count[d] += freq[multiple]

    # Process each element in A
    ans = []
    for x in A:
        best = 0
        # Iterate divisors in reverse order (descending)
        for d in reversed(divisors[x]):
            if count[d] >= K:
                best = d
                break
        ans.append(best)

    # Output the results
    sys.stdout.write('
'.join(map(str, ans)) + '
')

if __name__ == "__main__":
    main()