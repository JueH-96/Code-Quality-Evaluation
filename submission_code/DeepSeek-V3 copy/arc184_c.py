def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # The maximum value of f(i) is the maximum number of 1s in the binary representation of (i + A_k) for any i
    # Since the creases are determined by the binary representation of their positions, the number of mountain folds is the number of 1s in the binary representation of the position
    # To maximize f(i), we need to find the maximum number of positions (i + A_k) that have the same number of 1s in their binary representation
    
    # The number of 1s in the binary representation of a number is called its Hamming weight
    # The maximum Hamming weight for a number up to 2^100 is 100
    
    # We need to find the maximum number of A_k such that (i + A_k) has the same Hamming weight for some i
    
    # Since i can be any number, we can consider the Hamming weight of A_k itself, but shifted by i
    
    # However, it's easier to consider that the maximum number of A_k that can have the same Hamming weight when added to i is the maximum count of A_k that have the same Hamming weight modulo 2
    
    # But a better approach is to realize that the maximum f(i) is the maximum number of A_k that have the same parity (odd or even) in their Hamming weight
    
    # So, we count the number of A_k with even Hamming weight and the number with odd Hamming weight, and take the maximum
    
    # Function to count the number of 1s in the binary representation of a number
    def hamming_weight(x):
        return bin(x).count('1')
    
    even_count = 0
    odd_count = 0
    for a in A:
        if hamming_weight(a) % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
    
    print(max(even_count, odd_count))

if __name__ == "__main__":
    main()