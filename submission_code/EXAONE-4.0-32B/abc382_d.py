import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    
    results = []
    
    def dfs(seq):
        if len(seq) == N:
            results.append(seq)
            return
        
        L = len(seq)
        if L == 0:
            low_bound = 1
            high_bound = M - 10 * (N - 1)
        else:
            low_bound = seq[-1] + 10
            high_bound = M - 10 * (N - L - 1)
        
        if low_bound > high_bound:
            return
        
        for a in range(low_bound, high_bound + 1):
            dfs(seq + [a])
            
    dfs([])
    
    print(len(results))
    for seq in results:
        print(" ".join(map(str, seq)))

if __name__ == "__main__":
    main()