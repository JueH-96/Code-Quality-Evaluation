def can_rearrange_subsequences(N, Q, A, B, queries):
    from collections import Counter
    results = []
    
    for l, r, L, R in queries:
        # Adjust indices to be zero-based
        l -= 1
        r -= 1
        L -= 1
        R -= 1
        
        # Get the subsequences
        sub_A = A[l:r + 1]
        sub_B = B[L:R + 1]
        
        # Count the occurrences of each number in both subsequences
        count_A = Counter(sub_A)
        count_B = Counter(sub_B)
        
        # Compare the counts
        if count_A == count_B:
            results.append("Yes")
        else:
            results.append("No")
    
    return results

import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    
    A = list(map(int, data[2:N + 2]))
    B = list(map(int, data[N + 2:2 * N + 2]))
    
    queries = []
    index = 2 * N + 2
    for _ in range(Q):
        l = int(data[index])
        r = int(data[index + 1])
        L = int(data[index + 2])
        R = int(data[index + 3])
        queries.append((l, r, L, R))
        index += 4
    
    results = can_rearrange_subsequences(N, Q, A, B, queries)
    print("
".join(results))

if __name__ == "__main__":
    main()