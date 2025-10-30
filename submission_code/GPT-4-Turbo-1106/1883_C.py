import sys
from collections import Counter

def min_operations_to_divisible(test_cases):
    results = []
    for n, k, a in test_cases:
        count = Counter(a_i % k for a_i in a)
        ops = 0
        for remainder, freq in count.items():
            if remainder == 0:
                continue
            complement = (k - remainder) % k
            needed = (freq * remainder) % k
            if needed == 0:
                continue
            ops += (needed + complement - 1) // complement
        results.append(ops)
    return results

def main():
    t = int(input().strip())
    test_cases = []
    for _ in range(t):
        n, k = map(int, input().strip().split())
        a = list(map(int, input().strip().split()))
        test_cases.append((n, k, a))
    results = min_operations_to_divisible(test_cases)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()