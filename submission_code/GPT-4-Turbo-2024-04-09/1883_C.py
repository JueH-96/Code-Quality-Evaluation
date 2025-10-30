def min_operations_to_make_product_divisible_by_k(t, test_cases):
    results = []
    for n, k, a in test_cases:
        # Count residues modulo k
        count_mod = [0] * k
        for number in a:
            count_mod[number % k] += 1
        
        # Calculate the number of operations needed
        operations = 0
        
        # Check for each residue
        for residue in range(k):
            if residue == 0:
                continue  # No need to change if already divisible by k
            needed = (k - residue) % k
            while count_mod[residue] > 0:
                # How many times we can pair this residue with its complement
                if count_mod[needed] > 0:
                    # Pair them directly
                    count_mod[residue] -= 1
                    count_mod[needed] -= 1
                else:
                    # We need to increment this residue to make it divisible by k
                    operations += needed
                    count_mod[residue] -= 1
        
        results.append(operations)
    
    return results

import sys
input = sys.stdin.read
data = input().split()
index = 0
t = int(data[index])
index += 1
test_cases = []

for _ in range(t):
    n = int(data[index])
    k = int(data[index + 1])
    index += 2
    a = list(map(int, data[index:index + n]))
    index += n
    test_cases.append((n, k, a))

results = min_operations_to_make_product_divisible_by_k(t, test_cases)
for result in results:
    print(result)