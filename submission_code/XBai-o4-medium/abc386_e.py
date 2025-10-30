import itertools

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    K = int(input[idx+1])
    idx += 2
    A = list(map(int, input[idx:idx+N]))
    
    max_xor = 0
    
    for combo in itertools.combinations(A, K):
        xor_val = 0
        for num in combo:
            xor_val ^= num
        if xor_val > max_xor:
            max_xor = xor_val
    
    print(max_xor)

if __name__ == "__main__":
    main()