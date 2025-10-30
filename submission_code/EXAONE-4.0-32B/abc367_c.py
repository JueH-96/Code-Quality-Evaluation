import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    k = int(data[1])
    R = list(map(int, data[2:2+n]))
    
    results = []
    current = []
    
    def dfs(pos, s_mod):
        if pos == n:
            if s_mod % k == 0:
                results.append(tuple(current))
            return
        
        for num in range(1, R[pos] + 1):
            current.append(num)
            new_mod = (s_mod + num) % k
            dfs(pos + 1, new_mod)
            current.pop()
            
    dfs(0, 0)
    
    for seq in results:
        print(" ".join(map(str, seq)))
        
if __name__ == "__main__":
    main()