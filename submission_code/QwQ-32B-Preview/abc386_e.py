import itertools

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    if K == 1:
        print(max(A))
    else:
        max_xor = 0
        for combo in itertools.combinations(A, K):
            current_xor = combo[0]
            for num in combo[1:]:
                current_xor ^= num
            if current_xor > max_xor:
                max_xor = current_xor
        print(max_xor)

if __name__ == "__main__":
    main()