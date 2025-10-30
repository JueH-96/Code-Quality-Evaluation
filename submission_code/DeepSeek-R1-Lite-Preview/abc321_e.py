def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        X = int(data[index + 1])
        K = int(data[index + 2])
        index += 3
        
        if N == 0:
            results.append(0)
            continue
        
        if X > N:
            results.append(0)
            continue
        
        h = N.bit_length()  # Height of the tree
        
        def depth(u):
            return u.bit_length() - 1 if u > 0 else 0
        
        def count_nodes_at_depth(u, k):
            target_depth = depth(u) + k
            if target_depth < depth(u) or target_depth >= h:
                return 0
            left = u * (1 << k)
            right = left + (1 << k) - 1
            if left > N:
                return 0
            return min(N, right) - left + 1
        
        depth_X = depth(X)
        max_t = min(K, depth_X)
        count = 0
        
        for t in range(0, max_t + 1):
            if t == 0:
                A_t = X
                count_nodes_A_t = count_nodes_at_depth(X, K)
                count += count_nodes_A_t
            else:
                A_t = X >> t
                target_depth = depth_X + K - 2 * t
                if target_depth < (depth_X - t) or target_depth >= h:
                    continue
                count_nodes_A_t = count_nodes_at_depth(A_t, K - t)
                if target_depth >= depth_X:
                    target_sub_depth = target_depth - depth_X
                    count_nodes_X = count_nodes_at_depth(X, target_sub_depth)
                else:
                    count_nodes_X = 0
                count += count_nodes_A_t - count_nodes_X
        
        results.append(str(count))
    
    print("
".join(results))

if __name__ == '__main__':
    main()