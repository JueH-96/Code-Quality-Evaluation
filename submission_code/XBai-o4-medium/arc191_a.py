def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    S = list(input[idx])
    idx += 1
    T = input[idx]
    
    # Precompute max_suffix for T
    max_suffix = [0] * M
    max_suffix[-1] = int(T[-1])
    for i in range(M-2, -1, -1):
        max_suffix[i] = max(int(T[i]), max_suffix[i+1])
    
    left = 0
    right = N - 1
    
    for k in range(M):
        t_char = T[k]
        t = int(t_char)
        if t == max_suffix[k]:
            # Key step
            while left < N and int(S[left]) >= t:
                left += 1
            if left < N:
                S[left] = t_char
                left += 1
            else:
                # Replace last character
                S[-1] = t_char
        else:
            # Non-key step
            while right >= 0 and int(S[right]) >= t:
                right -= 1
            if right >= 0:
                S[right] = t_char
                right -= 1
            else:
                # Replace last character
                S[-1] = t_char
    
    print(''.join(S))

if __name__ == '__main__':
    main()