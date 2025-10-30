def main():
    import sys
    data = sys.stdin.read().splitlines()
    N = int(data[0].strip())
    A = []
    B = []
    for i in range(1, 1 + N):
        a, b = map(int, data[i].split())
        A.append(a)
        B.append(b)
    
    total_mask = 1 << N
    dp = [False] * total_mask
    dp[0] = False
    
    for mask in range(1, total_mask):
        bits = []
        for i in range(N):
            if mask & (1 << i):
                bits.append(i)
        k = len(bits)
        if k < 2:
            dp[mask] = False
        else:
            found_win = False
            for i_idx in range(len(bits)):
                for j_idx in range(i_idx + 1, len(bits)):
                    card1 = bits[i_idx]
                    card2 = bits[j_idx]
                    if A[card1] == A[card2] or B[card1] == B[card2]:
                        next_mask = mask ^ (1 << card1) ^ (1 << card2)
                        if not dp[next_mask]:
                            found_win = True
                            break
                if found_win:
                    break
            dp[mask] = found_win
            
    if dp[total_mask - 1]:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()