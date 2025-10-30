def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    if len(data) < 3:
        print("! 0")
        return
    N = int(data[0])
    L = int(data[1])
    R = int(data[2])
    
    sum_ans = 0
    current = L
    while current <= R:
        if current == 0:
            max_s = R + 1
            current_size = 1 << (max_s.bit_length() - 1)
        else:
            current_size = current & -current
            max_possible = R - current + 1
            while current_size > max_possible:
                current_size >>= 1
        
        # Now query i and j
        if current_size == 0:
            break  # This case shouldn't occur as per problem constraints
        i = (current_size).bit_length() - 1
        j = current // current_size
        
        # Ensure j is within valid bounds (though problem states it's valid)
        # Output the query
        print(f"? {i} {j}", flush=True)
        # Read response
        t = int(sys.stdin.readline())
        if t == -1:
            print("! 0")
            return
        sum_ans = (sum_ans + t) % 100
        current += current_size
    
    print(f"! {sum_ans % 100}")

if __name__ == "__main__":
    main()