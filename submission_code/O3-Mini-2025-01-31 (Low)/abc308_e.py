def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    S = data[1+n]

    # Precompute mex for all triples (x,y,z) where x,y,z are in {0,1,2}
    mex_table = {}
    for x in range(3):
        for y in range(3):
            for z in range(3):
                present = {x, y, z}
                m = 0
                while m in present:
                    m += 1
                mex_table[(x, y, z)] = m

    # right_counts will maintain the frequency of A[k] for indices k where S[k]=='X'
    right_counts = [0, 0, 0]  # counts for values 0, 1, 2
    for i in range(n):
        if S[i] == 'X':
            right_counts[A[i]] += 1

    # m_counts will maintain frequency for A[i] where S[i]=='M' and i < current j
    m_counts = [0, 0, 0]

    ans = 0
    # We iterate over positions (j from 0 to n-1) and consider those with S[j]=='E'
    # such that i < j < k where S[i]=='M' and S[k]=='X'.
    for j in range(n):
        # If the current position is 'X', remove it from right_counts,
        # because we need k > j.
        if S[j] == 'X':
            right_counts[A[j]] -= 1

        if S[j] == 'E':
            e_val = A[j]
            # Sum over all possible values for A[i] from left and A[k] from right.
            for a in range(3):
                if m_counts[a] == 0:
                    continue
                for c in range(3):
                    if right_counts[c] == 0:
                        continue
                    # Compute mex for the triple (A[i], A[j], A[k]) = (a, e_val, c)
                    m_val = mex_table[(a, e_val, c)]
                    ans += m_counts[a] * right_counts[c] * m_val

        # If the current position is 'M', then update m_counts for future indices j.
        if S[j] == 'M':
            m_counts[A[j]] += 1

    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()