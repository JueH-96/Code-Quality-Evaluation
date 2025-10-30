import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx +=1
    Q = int(data[idx])
    idx +=1
    
    parent = list(range(N+2))  # 1-based
    rank = [1]*(N+2)
    color = list(range(N+2))  # color[i] is the color of the component whose root is i
    min_idx = list(range(N+2))
    max_idx = list(range(N+2))
    size = [1]*(N+2)
    cnt = [0]*(N+2)
    for i in range(1, N+1):
        cnt[i] = 1
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    output = []
    for _ in range(Q):
        query = data[idx]
        idx +=1
        if query == '1':
            x = int(data[idx])
            idx +=1
            c = int(data[idx])
            idx +=1
            root = find(x)
            if color[root] == c:
                continue
            current_color = color[root]
            s = size[root]
            cnt[current_color] -= s
            cnt[c] += s
            color[root] = c
            l = min_idx[root] -1
            if l >=1:
                left_root = find(l)
                if color[left_root] == c:
                    if root != left_root:
                        if rank[root] > rank[left_root]:
                            parent[left_root] = root
                            if min_idx[root] > min_idx[left_root]:
                                min_idx[root] = min_idx[left_root]
                            if max_idx[root] < max_idx[left_root]:
                                max_idx[root] = max_idx[left_root]
                            size[root] += size[left_root]
                        else:
                            parent[root] = left_root
                            if min_idx[left_root] > min_idx[root]:
                                min_idx[left_root] = min_idx[root]
                            if max_idx[left_root] < max_idx[root]:
                                max_idx[left_root] = max_idx[root]
                            size[left_root] += size[root]
                            if rank[root] == rank[left_root]:
                                rank[left_root] +=1
            r = max_idx[root] +1
            if r <= N:
                right_root = find(r)
                if color[right_root] == c:
                    if root != right_root:
                        if rank[root] > rank[right_root]:
                            parent[right_root] = root
                            if min_idx[root] > min_idx[right_root]:
                                min_idx[root] = min_idx[right_root]
                            if max_idx[root] < max_idx[right_root]:
                                max_idx[root] = max_idx[right_root]
                            size[root] += size[right_root]
                        else:
                            parent[root] = right_root
                            if min_idx[right_root] > min_idx[root]:
                                min_idx[right_root] = min_idx[root]
                            if max_idx[right_root] < max_idx[root]:
                                max_idx[right_root] = max_idx[root]
                            size[right_root] += size[root]
                            if rank[root] == rank[right_root]:
                                rank[right_root] +=1
        else:
            c = int(data[idx])
            idx +=1
            output.append(str(cnt[c]))
    
    print('
'.join(output))

if __name__ == '__main__':
    main()