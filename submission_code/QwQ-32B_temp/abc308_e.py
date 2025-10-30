import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    S = sys.stdin.readline().strip()

    # Precompute mex table
    mex_table = [[[0]*3 for _ in range(3)] for __ in range(3)]
    for a in range(3):
        for b in range(3):
            for c in range(3):
                s = {a, b, c}
                mex = 0
                while mex in s:
                    mex += 1
                mex_table[a][b][c] = mex

    # Precompute m_before
    m_before = [[0]*3 for _ in range(N+1)]
    current = [0, 0, 0]
    m_before[0] = current.copy()
    for i in range(N):
        if S[i] == 'M':
            current[A[i]] += 1
        m_before[i+1] = current.copy()

    # Precompute x_after
    x_after = [[0]*3 for _ in range(N)]
    current_x = [0, 0, 0]
    for i in range(N-1, -1, -1):
        x_after[i] = current_x.copy()
        if S[i] == 'X':
            current_x[A[i]] += 1

    total = 0
    for j in range(N):
        if S[j] == 'E':
            b = A[j]
            m_counts = m_before[j]
            x_counts = x_after[j]
            for a in range(3):
                for c in range(3):
                    count = m_counts[a] * x_counts[c]
                    total += count * mex_table[a][b][c]
    print(total)

if __name__ == "__main__":
    main()