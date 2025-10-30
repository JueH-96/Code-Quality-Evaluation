def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    A = list(map(int, data[1].split()))
    S = data[2].strip()
    
    mex_table = {}
    for a in range(3):
        for b in range(3):
            for c in range(3):
                s = {a, b, c}
                mex_val = 0
                while mex_val in s:
                    mex_val += 1
                mex_table[(a, b, c)] = mex_val

    total_x = [0, 0, 0]
    for i in range(n):
        if S[i] == 'X':
            total_x[A[i]] += 1
            
    m_count = [0, 0, 0]
    x_remaining = total_x[:]
    ans = 0
    
    for j in range(n):
        if S[j] == 'X':
            x_remaining[A[j]] -= 1
        elif S[j] == 'E':
            a_val = A[j]
            for m_val in range(3):
                for x_val in range(3):
                    cnt = m_count[m_val] * x_remaining[x_val]
                    if cnt:
                        key = (m_val, a_val, x_val)
                        ans += cnt * mex_table[key]
        elif S[j] == 'M':
            m_count[A[j]] += 1
            
    print(ans)

if __name__ == "__main__":
    main()