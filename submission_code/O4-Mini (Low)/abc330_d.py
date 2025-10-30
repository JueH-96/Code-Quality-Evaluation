import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1:]
    # Count o's in each column
    col_cnt = [0]*N
    for i in range(N):
        row = S[i]
        for j,ch in enumerate(row):
            if ch=='o':
                col_cnt[j]+=1

    ans = 0
    # For each row, let its o‐columns be c1...ck
    # Each unordered pair (ci,cj) contributes:
    #   (col_cnt[ci]-1) + (col_cnt[cj]-1)
    # Summing over all pairs gives:
    #   (k-1)*sum(col_cnt[ci]) - k*(k-1)
    for i in range(N):
        cols = [j for j,ch in enumerate(S[i]) if ch=='o']
        k = len(cols)
        if k<2: 
            continue
        s = 0
        for c in cols:
            s += col_cnt[c]
        # Total contribution from this row
        ans += (k-1)*s - k*(k-1)

    print(ans)

if __name__=='__main__':
    main()