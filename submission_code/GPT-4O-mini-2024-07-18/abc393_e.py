import sys
from collections import defaultdict
from math import gcd
from functools import reduce

def max_gcd_with_k_elements(N, K, A):
    # Step 1: Count the frequency of each number in A
    freq = defaultdict(int)
    for number in A:
        freq[number] += 1

    # Step 2: Prepare a list of unique elements and their counts
    unique_elements = list(freq.keys())
    
    # Step 3: Prepare an array to store the maximum GCD for each element
    max_gcds = [0] * N
    
    # Step 4: Iterate through each possible GCD value from the maximum down to 1
    for g in range(1, 10**6 + 1):
        # Count how many elements are multiples of g
        count = 0
        for multiple in range(g, 10**6 + 1, g):
            count += freq[multiple]
        
        # If we have enough multiples of g, we can consider it
        if count >= K:
            # Now we can assign this GCD to the elements that are multiples of g
            for i in range(N):
                if A[i] % g == 0:
                    max_gcds[i] = max(max_gcds[i], g)
    
    return max_gcds

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:N + 2]))
    
    results = max_gcd_with_k_elements(N, K, A)
    
    # Print results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()