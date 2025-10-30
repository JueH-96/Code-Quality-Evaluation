import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    W = int(input[ptr])
    ptr += 1
    dp = [-float('inf')] * (W + 1)
    dp[0] = 0
    for _ in range(N):
        w = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        prev_dp = list(dp)
        for j in range(W, w - 1, -1):
            m_max = j // w
            l = 0
            r = m_max
            for _ in range(100):
                if l > r:
                    break
                m1 = l + (r - l) // 3
                m2 = r - (r - l) // 3
                val1 = -float('inf')
                if j - m1 * w >= 0:
                    val1 = prev_dp[j - m1 * w] + m1 * v - m1 * m1
                val2 = -float('inf')
                if j - m2 * w >= 0:
                    val2 = prev_dp[j - m2 * w] + m2 * v - m2 * m2
                if val1 < val2:
                    l = m1 + 1
                else:
                    r = m2 - 1
            best_val = -float('inf')
            for m_candidate in range(max(0, l - 2), min(m_max, r + 2) + 1):
                current_j = j - m_candidate * w
                if current_j >= 0:
                    current_val = prev_dp[current_j] + m_candidate * v - m_candidate * m_candidate
                    if current_val > best_val:
                        best_val = current_val
            if best_val > dp[j]:
                dp[j] = best_val
    print(int(max(dp)))

if __name__ == '__main__':
    main()