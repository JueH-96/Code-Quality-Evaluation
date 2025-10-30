def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix[i] = prefix[i - 1] ^ A[i - 1]
    
    total = 0
    for bit in range(32):
        set_count = 0
        for b in range(2, N + 1):
            # Number of a <= b-2
            num_a = b - 1
            if (prefix[b] >> bit) & 1:
                total += set_count
            else:
                total += (num_a - set_count)
            # Update set_count with prefix[b-2]
            if b - 2 >= 0 and (prefix[b - 2] >> bit) & 1:
                set_count += 1
    print(total)

if __name__ == "__main__":
    main()