import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] ^ A[i]
    
    ans = 0
    for b in range(31):
        zeros = 0
        ones = 0
        total_bit = 0
        prev_bit = (prefix[0] >> b) & 1
        for y in range(1, N + 1):
            current_bit = (prefix[y] >> b) & 1
            if current_bit == 0:
                total_bit += ones
            else:
                total_bit += zeros
            # Update zeros and ones based on prev_bit (bit of prefix[y-1])
            if prev_bit == 0:
                new_zeros = zeros + 1
                new_ones = ones
            else:
                new_zeros = zeros
                new_ones = ones + 1
            zeros, ones = new_zeros, new_ones
            prev_bit = current_bit
        ans += total_bit * (1 << b)
    print(ans)

if __name__ == "__main__":
    main()