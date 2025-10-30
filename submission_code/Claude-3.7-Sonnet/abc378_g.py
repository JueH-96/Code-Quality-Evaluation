import bisect
from itertools import permutations

def count_valid_permutations(A, B, M):
    perm_length = A * B - 1
    count = 0
    
    for perm in permutations(range(1, perm_length + 1)):
        # Calculate LIS using patience sort
        lis = []
        lis_dp = [0] * perm_length
        for i, num in enumerate(perm):
            pos = bisect.bisect_left(lis, num)
            if pos == len(lis):
                lis.append(num)
            else:
                lis[pos] = num
            lis_dp[i] = pos + 1
        
        if max(lis_dp) != A:
            continue
        
        # Calculate LDS using patience sort
        lds = []
        lds_dp = [0] * perm_length
        for i, num in enumerate(perm):
            pos = bisect.bisect_left(lds, -num)
            if pos == len(lds):
                lds.append(-num)
            else:
                lds[pos] = -num
            lds_dp[i] = pos + 1
        
        if max(lds_dp) != B:
            continue
        
        # Find the smallest last element of all increasing subsequences of length A
        min_last_element_lis = float('inf')
        for i in range(perm_length):
            if lis_dp[i] == A:
                min_last_element_lis = min(min_last_element_lis, perm[i])
        
        # Find the largest last element of all decreasing subsequences of length B
        max_last_element_lds = -float('inf')
        for i in range(perm_length):
            if lds_dp[i] == B:
                max_last_element_lds = max(max_last_element_lds, perm[i])
        
        # Check if there's a valid n
        if max_last_element_lds < min_last_element_lis:
            count = (count + 1) % M
    
    return count

def main():
    A, B, M = map(int, input().split())
    result = count_valid_permutations(A, B, M)
    print(result)

if __name__ == "__main__":
    main()