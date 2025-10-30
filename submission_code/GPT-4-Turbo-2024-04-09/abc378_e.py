def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))
    
    # Prefix sum and its mod
    prefix_mod = [0] * (N + 1)
    
    for i in range(1, N + 1):
        prefix_mod[i] = (prefix_mod[i - 1] + A[i - 1]) % M
    
    # Dictionary to count occurrences of each mod value
    mod_count = {}
    total_sum = 0
    
    for mod_value in prefix_mod:
        if mod_value in mod_count:
            total_sum += mod_count[mod_value]
            mod_count[mod_value] += 1
        else:
            mod_count[mod_value] = 1
    
    print(total_sum)

if __name__ == "__main__":
    main()