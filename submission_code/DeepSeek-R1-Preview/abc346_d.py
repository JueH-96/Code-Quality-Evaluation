import sys

def main():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    C = list(map(int, sys.stdin.readline().split()))
    
    # Precompute cost0 and cost1 for each position j and x in {0, 1}
    # cost0[j][x] is the cost to set T[j] to x
    # cost1[j][x] is the cost to set T[j] to 1 - x
    cost0 = [[0]*2 for _ in range(N)]
    cost1 = [[0]*2 for _ in range(N)]
    for j in range(N):
        s = int(S[j])
        for x in [0, 1]:
            cost0[j][x] = 0 if s == x else C[j]
            cost1[j][x] = 0 if s == (1 - x) else C[j]
    
    min_total = float('inf')
    
    for x in [0, 1]:
        # Initialize prefix sums
        even_0 = [0] * (N + 1)
        even_1 = [0] * (N + 1)
        odd_0 = [0] * (N + 1)
        odd_1 = [0] * (N + 1)
        
        for j in range(1, N + 1):
            idx = j - 1  # 0-based index
            if j % 2 == 0:
                even_0[j] = even_0[j-1] + cost0[idx][x]
                even_1[j] = even_1[j-1] + cost1[idx][x]
                odd_0[j] = odd_0[j-1]
                odd_1[j] = odd_1[j-1]
            else:
                odd_0[j] = odd_0[j-1] + cost0[idx][x]
                odd_1[j] = odd_1[j-1] + cost1[idx][x]
                even_0[j] = even_0[j-1]
                even_1[j] = even_1[j-1]
        
        # Initialize suffix sums
        suffix_even_0 = [0] * (N + 2)
        suffix_even_1 = [0] * (N + 2)
        suffix_odd_0 = [0] * (N + 2)
        suffix_odd_1 = [0] * (N + 2)
        
        for j in range(N, 0, -1):
            idx = j - 1  # 0-based index
            if j % 2 == 0:
                suffix_even_0[j] = suffix_even_0[j+1] + cost0[idx][x]
                suffix_even_1[j] = suffix_even_1[j+1] + cost1[idx][x]
                suffix_odd_0[j] = suffix_odd_0[j+1]
                suffix_odd_1[j] = suffix_odd_1[j+1]
            else:
                suffix_odd_0[j] = suffix_odd_0[j+1] + cost0[idx][x]
                suffix_odd_1[j] = suffix_odd_1[j+1] + cost1[idx][x]
                suffix_even_0[j] = suffix_even_0[j+1]
                suffix_even_1[j] = suffix_even_1[j+1]
        
        # Iterate through all possible i (1-based, from 1 to N-1)
        for i in range(1, N):
            # Compute left part sum
            i_parity = i % 2
            if i_parity == 0:
                sum_left = even_0[i] + odd_1[i]
            else:
                sum_left = even_1[i] + odd_0[i]
            
            # Compute right part sum
            i_plus_1 = i + 1
            i_plus_1_parity = i_plus_1 % 2
            if i_plus_1_parity == 0:
                sum_right = suffix_even_0[i_plus_1] + suffix_odd_1[i_plus_1]
            else:
                sum_right = suffix_even_1[i_plus_1] + suffix_odd_0[i_plus_1]
            
            total = sum_left + sum_right
            if total < min_total:
                min_total = total
    
    print(min_total)

if __name__ == '__main__':
    main()