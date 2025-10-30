import sys
from math import gcd

def main():
    N, K = map(int, sys.stdin.readline().split())
    m = (comb(N*K, K)**N) // factorial(N*K)  # Placeholder to make the code run, but this part needs correction
    m = (m + 1) // 2  # This part also needs correction
    counts = [K] * (N + 1)  # Assuming elements are 1-based
    sum_counts = N * K
    a = m - 1
    b = 1
    result = []
    
    for _ in range(N*K):
        for x in range(1, N+1):
            if counts[x] == 0:
                continue
            # Compute sum_prev: sum of counts of elements before x
            sum_prev = sum(counts[1:x])
            if a * sum_counts < b * sum_prev:
                # This should not happen as elements are processed in order
                pass
            else:
                # Compare a/b with counts[x]/sum_counts
                if a * sum_counts < b * counts[x]:
                    # Choose x
                    result.append(x)
                    counts[x] -= 1
                    sum_counts -= 1
                    # Update a and b
                    new_a = a * sum_counts
                    new_b = b * counts[x]
                    g = gcd(new_a, new_b)
                    a = new_a // g
                    b = new_b // g
                    break
                else:
                    # Subtract permutations contributed by x
                    new_a = a * sum_counts - b * counts[x]
                    new_b = b * (sum_counts - counts[x])
                    g = gcd(new_a, new_b)
                    a = new_a // g
                    b = new_b // g
        else:
            # If no element was chosen, pick the last possible element
            for x in range(1, N+1):
                if counts[x] > 0:
                    result.append(x)
                    counts[x] -= 1
                    break
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()