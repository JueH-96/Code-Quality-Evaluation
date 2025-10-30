def main():
    import sys
    L, R = map(int, sys.stdin.readline().split())
    current = L
    result = []
    
    while current < R:
        max_possible = R - current
        if max_possible <= 0:
            break
        
        # Find the maximum i where 2^(i+1) <= max_possible
        max_i = 0
        while (1 << (max_i + 1)) <= max_possible:
            max_i += 1
        
        # Find the largest i where current is divisible by 2^i
        found_i = 0
        for i in range(max_i, -1, -1):
            if (current % (1 << i)) == 0:
                found_i = i
                break
        
        block_length = 1 << found_i
        end = current + block_length
        result.append((current, end))
        current = end
    
    print(len(result))
    for l, r in result:
        print(l, r)

if __name__ == "__main__":
    main()