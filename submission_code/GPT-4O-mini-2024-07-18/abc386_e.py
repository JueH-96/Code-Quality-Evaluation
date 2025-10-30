def max_xor_of_k_elements(N, K, A):
    from itertools import combinations

    max_xor = 0
    for combo in combinations(A, K):
        current_xor = 0
        for number in combo:
            current_xor ^= number
        max_xor = max(max_xor, current_xor)

    return max_xor

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2 + N]))
    
    result = max_xor_of_k_elements(N, K, A)
    print(result)