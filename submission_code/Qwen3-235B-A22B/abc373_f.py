import sys

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    W = int(input[idx])
    idx +=1
    items = []
    for _ in range(N):
        w = int(input[idx])
        v = int(input[idx+1])
        items.append( (w, v) )
        idx +=2
    
    dp = [ -float('inf') ] * (W + 1)
    dp[0] = 0
    
    for (w, v) in items:
        new_dp = dp[:]
        for j in range(W +1):
            max_k = min( j // w, 200 )
            for k in range(1, max_k+1):
                prev = j - k * w
                if prev >=0 and dp[prev] != -float('inf'):
                    new_val = dp[prev] + v * k - k *k
                    if new_val > new_dp[j]:
                        new_dp[j] = new_val
        dp = new_dp
    
    print( max(dp) )

if __name__ == "__main__":
    main()