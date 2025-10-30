def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N
    S = input[idx]
    
    # Precompute prefix_m_counts
    prefix_m_counts = [(0, 0, 0)]
    for i in range(N):
        current = list(prefix_m_counts[-1])
        if S[i] == 'M':
            a_val = A[i]
            current[a_val] += 1
        prefix_m_counts.append(tuple(current))
    
    # Precompute suffix_x
    suffix_x = [(0, 0, 0) for _ in range(N)]
    current_x0 = 0
    current_x1 = 0
    current_x2 = 0
    for i in range(N-1, -1, -1):
        suffix_x[i] = (current_x0, current_x1, current_x2)
        if S[i] == 'X':
            a_val = A[i]
            if a_val == 0:
                current_x0 += 1
            elif a_val == 1:
                current_x1 += 1
            else:
                current_x2 += 1
    
    total_sum = 0
    
    for j in range(N):
        if S[j] == 'E':
            m0, m1, m2 = prefix_m_counts[j]
            x0, x1, x2 = suffix_x[j]
            aj = A[j]
            contrib = 0
            for a in [0, 1, 2]:
                for b in [0, 1, 2]:
                    ma = [m0, m1, m2][a]
                    mb = [x0, x1, x2][b]
                    count = ma * mb
                    if count == 0:
                        continue
                    # compute mex
                    s = {a, aj, b}
                    if 0 not in s:
                        mex = 0
                    elif 1 not in s:
                        mex = 1
                    elif 2 not in s:
                        mex = 2
                    else:
                        mex = 3
                    contrib += count * mex
            total_sum += contrib
    
    print(total_sum)

if __name__ == "__main__":
    main()