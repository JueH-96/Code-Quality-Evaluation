def main():
    import sys

    # Read N, L, R from input
    input_line = sys.stdin.readline().strip()
    N_L_R = input_line.split()
    while len(N_L_R) < 3:
        input_line += ' ' + sys.stdin.readline().strip()
        N_L_R = input_line.split()
    N, L, R = map(int, N_L_R)

    total_sum = 0
    current_L = L
    current_R = R

    # Iterate from highest level to lowest
    for i in range(N-1, -1, -1):
        block_size = 1 << i  # 2^i
        while current_L + block_size <= current_R:
            j = current_L // block_size
            # Check if the block starts exactly at current_L
            if j * block_size != current_L:
                # Move to the next block that starts after current_L
                j = (current_L + block_size - 1) // block_size
                current_L = j * block_size
                continue
            # Ask for the sum of this block
            print(f"? {i} {j}")
            sys.stdout.flush()
            T = int(sys.stdin.readline().strip())
            if T == -1:
                return
            total_sum = (total_sum + T) % 100
            current_L += block_size

    # Handle any remaining elements
    while current_L < current_R:
        j = current_L
        print(f"? 0 {j}")
        sys.stdout.flush()
        T = int(sys.stdin.readline().strip())
        if T == -1:
            return
        total_sum = (total_sum + T) % 100
        current_L += 1

    # Output the result
    print(f"! {total_sum}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()