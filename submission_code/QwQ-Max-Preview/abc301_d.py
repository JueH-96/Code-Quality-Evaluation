def main():
    import sys
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline())
    len_S = len(S)
    max_suffix = [0] * (len_S + 1)
    
    for i in range(len_S - 1, -1, -1):
        current = S[i]
        shift = len_S - 1 - i
        if current == '?':
            bit = 1
        else:
            bit = int(current)
        max_suffix[i] = (bit << shift) + max_suffix[i + 1]
    
    if max_suffix[0] <= N:
        print(max_suffix[0])
        return
    
    result = 0
    for i in range(len_S):
        current_char = S[i]
        remaining_max = max_suffix[i + 1]
        shift = len_S - 1 - i
        if current_char == '?':
            temp_candidate = (1 << shift) + remaining_max
            if temp_candidate <= N:
                result += (1 << shift)
        else:
            current_bit = int(current_char)
            candidate = (current_bit << shift) + remaining_max
            if candidate > N:
                print(-1)
                return
            else:
                result += current_bit << shift
    print(result)

if __name__ == "__main__":
    main()