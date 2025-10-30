def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # The maximum value of f(i) is the maximum number of 1s in the binary representation of (i + A_k) for any i
    # Since the creases are determined by the binary representation of their positions, we need to find the maximum number of A_k that have the same parity in their binary representation when added to i
    
    # The key insight is that the parity of the number of 1s in the binary representation of (i + A_k) is the same as the parity of the number of 1s in i and A_k combined
    # However, to maximize f(i), we need to find the maximum number of A_k that have the same parity in their binary representation when added to i
    
    # Since the creases are determined by the binary representation, the maximum number of A_k that can have the same parity in their binary representation when added to i is the maximum number of A_k that have the same parity in their binary representation
    
    # So, we can count the number of A_k that have an even number of 1s and the number that have an odd number of 1s, and take the maximum of these two counts
    
    even_count = 0
    odd_count = 0
    
    for a in A:
        if bin(a).count('1') % 2 == 0:
            even_count += 1
        else:
            odd_count += 1
    
    print(max(even_count, odd_count))

if __name__ == "__main__":
    main()