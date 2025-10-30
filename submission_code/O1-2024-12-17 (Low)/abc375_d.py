def main():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    
    # Positions of each character A-Z (26 lists).
    # We'll store 1-based indices of their occurrences.
    positions = [[] for _ in range(26)]
    
    for i, ch in enumerate(S):
        positions[ord(ch) - ord('A')].append(i+1)
    
    ans = 0
    
    # Precompute answers for each character separately
    for pos in positions:
        m = len(pos)
        if m < 2:
            continue
        # Build prefix sums P and Q
        # P[i] = p[1] + p[2] + ... + p[i]
        # Q[i] = 1*p[1] + 2*p[2] + ... + i*p[i]
        P = [0]*(m+1)
        Q = [0]*(m+1)
        for i in range(m):
            P[i+1] = P[i] + pos[i]
            Q[i+1] = Q[i] + (i+1)*pos[i]
        
        # sum_{r<s} p_s = Q[m] - P[m]
        sumPs = Q[m] - P[m]
        # sum_{r<s} p_r = m*P[m] - Q[m]
        sumPr = m*P[m] - Q[m]
        # number of pairs
        pair_count = m*(m-1)//2
        # sum_{r<s} (p_s - p_r - 1)
        ans += sumPs - sumPr - pair_count
    
    print(ans)

if __name__ == "__main__":
    main()