def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    score_taka = 0
    score_aoki = 0
    
    idx = 1
    for _ in range(N):
        x = int(data[idx]); y = int(data[idx+1])
        score_taka += x
        score_aoki += y
        idx += 2
    
    if score_taka > score_aoki:
        print("Takahashi")
    elif score_aoki > score_taka:
        print("Aoki")
    else:
        print("Draw")

def main():
    solve()

if __name__ == "__main__":
    main()