def count_distinct_sequences(N, s):
    MOD = 998244353
    
    # Count the number of 1's in the string s
    count_ones = s.count('1')
    
    # If there are no 1's, the only sequence is (0, 0, ..., 0)
    if count_ones == 0:
        return 1
    
    # If there are 1's, we can form sequences based on the number of 1's
    # The in-degrees can vary based on how we direct the edges
    # Each 1 can connect to the vertices around it in different ways
    
    # The number of distinct sequences is given by the formula:
    # (count_ones + 1)^(N) - 1
    # We subtract 1 to exclude the all-zero sequence
    
    result = pow(count_ones + 1, N, MOD) - 1
    return result % MOD

import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    s = data[1]
    
    result = count_distinct_sequences(N, s)
    print(result)

if __name__ == "__main__":
    main()