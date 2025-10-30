def max_11_22_subsequence(N, Q, S, queries):
    results = []
    
    for L, R in queries:
        # Adjust for 0-based indexing
        L -= 1
        R -= 1
        
        # Count occurrences of '1', '2', and '/'
        count_1 = S[L:R+1].count('1')
        count_2 = S[L:R+1].count('2')
        count_slash = S[L:R+1].count('/')
        
        # To form an 11/22 string, we need:
        # - k '1's
        # - 1 '/' (which is the middle character)
        # - k '2's
        # The total length of the 11/22 string will be 2k + 1
        
        # The maximum k we can use is limited by the counts of '1', '2', and '/'
        max_k = min(count_1, count_2)
        
        if max_k > 0 and count_slash > 0:
            # The length of the longest 11/22 string we can form
            results.append(2 * max_k + 1)
        else:
            results.append(0)
    
    return results

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    N, Q = map(int, data[0].split())
    S = data[1]
    queries = [tuple(map(int, line.split())) for line in data[2:2 + Q]]
    
    results = max_11_22_subsequence(N, Q, S, queries)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()