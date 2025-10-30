import sys
import bisect
from collections import defaultdict

MOD = 998244353

def main():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    total_vertices = 2 * N
    balance = [0] * (total_vertices + 1)
    for i in range(total_vertices):
        if S[i] == 'W':
            balance[i+1] = balance[i] + 1
        else:
            balance[i+1] = balance[i] - 1

    # Preprocess positions for each balance value
    pos_dict = defaultdict(list)
    for i in range(total_vertices + 1):
        pos_dict[balance[i]].append(i)

    # Compute first_return array
    first_return = [-1] * total_vertices
    for start in range(total_vertices):
        target = balance[start]
        positions = pos_dict.get(target, [])
        # Find the smallest position in positions that is > start
        idx = bisect.bisect_right(positions, start)
        if idx < len(positions):
            end = positions[idx]
            first_return[start] = end - 1  # segment from start to end-1
        else:
            first_return[start] = -1

    # Precompute factorials up to N
    max_fact = N
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i % MOD

    # Initialize dp_val and m_total
    dp_val = [0] * (total_vertices + 2)
    m_total = [0] * (total_vertices + 2)

    for start in range(total_vertices - 1, -1, -1):
        if start >= total_vertices:
            dp_val[start] = 1
            m_total[start] = 0
        else:
            if first_return[start] == -1:
                length = total_vertices - start
                m_total[start] = length // 2
                dp_val[start] = fact[m_total[start]]
            else:
                end_pos = first_return[start]
                len_segment = end_pos - start + 1
                m1 = len_segment // 2
                remaining_start = end_pos + 1
                if remaining_start > total_vertices:
                    m_total_remaining = 0
                else:
                    m_total_remaining = m_total[remaining_start]
                m_total_current = m1 + m_total_remaining
                # Compute total_pairings
                if m_total_current > max_fact:
                    total_pairings = 0
                else:
                    total_pairings = fact[m_total_current]
                # Compute invalid_pairings
                if m1 > max_fact:
                    invalid = 0
                else:
                    invalid = fact[m1] * dp_val[remaining_start] % MOD
                dp_val[start] = (total_pairings - invalid) % MOD
                m_total[start] = m_total_current

    print(dp_val[0] % MOD)

if __name__ == "__main__":
    main()