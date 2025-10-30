import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = []
    B = []
    idx = 1
    for i in range(n):
        a = int(data[idx])
        b = int(data[idx + 1])
        idx += 2
        A.append(a)
        B.append(b)
    
    memo = {}
    
    def dfs(mask):
        if mask in memo:
            return memo[mask]
        for i in range(n):
            if mask & (1 << i):
                for j in range(i + 1, n):
                    if mask & (1 << j):
                        if A[i] == A[j] or B[i] == B[j]:
                            new_mask = mask ^ (1 << i) ^ (1 << j)
                            if not dfs(new_mask):
                                memo[mask] = True
                                return True
        memo[mask] = False
        return False
    
    initial_mask = (1 << n) - 1
    result = dfs(initial_mask)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()