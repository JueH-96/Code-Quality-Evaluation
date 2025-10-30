MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    A = list(map(int, data[1:n+1]))
    
    parent = list(range(n + 2))  # 1-based indices, parent[i] = i
    left = list(range(n + 2))
    right = list(range(n + 2))
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    ans = 1
    for i in range(n, 0, -1):
        a = A[i-1]
        l = a + 1
        r = i
        
        # Find the root of l and r
        root_l = find(l)
        root_r = find(r)
        
        # Merge all intervals that overlap with [l, r]
        # Find the leftmost and rightmost of the merged interval
        current_l = root_l
        current_r = root_r
        
        # Expand to the left
        while current_l > l:
            prev = current_l - 1
            prev_root = find(prev)
            if prev_root != prev:
                prev = find(prev)
            if prev < l:
                break
            current_l = prev
        
        # Expand to the right
        while current_r < r:
            next_root = find(current_r + 1)
            if next_root != current_r + 1:
                next_root = find(next_root)
            if next_root > r:
                break
            current_r = next_root
        
        # Update the parent, left, and right for the merged interval
        parent[current_l] = current_r
        left[current_r] = left[current_l]
        right[current_r] = right[current_r]
        left[current_l] = right[current_l] = current_r
        
        # Calculate the size of the merged interval
        size = right[current_r] - left[current_r] + 1
        ans = ans * size % MOD
    
    print(ans)

if __name__ == "__main__":
    main()