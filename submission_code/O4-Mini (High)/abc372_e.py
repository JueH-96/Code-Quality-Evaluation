import sys

def main():
    import sys
    input = sys.stdin.readline
    N, Q = map(int, input().split())
    
    # DSU parent and component size
    parent = list(range(N+1))
    comp_size = [1] * (N+1)
    
    # For each root, store up to 10 largest vertex numbers in its component
    topk = [[i] for i in range(N+1)]
    
    def find(x):
        # iterative with path halving
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    answers = []
    for _ in range(Q):
        line = input().split()
        t = line[0]
        if t == '1':
            # Add edge u-v => union their components
            u = int(line[1])
            v = int(line[2])
            ru = find(u)
            rv = find(v)
            if ru == rv:
                continue
            # union by size: attach smaller under larger
            if comp_size[ru] < comp_size[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            comp_size[ru] += comp_size[rv]
            # merge the two top-10 lists
            a = topk[ru]
            b = topk[rv]
            i = j = 0
            la, lb = len(a), len(b)
            merged = []
            # merge in descending order, cap at 10
            while len(merged) < 10 and i < la and j < lb:
                if a[i] > b[j]:
                    merged.append(a[i])
                    i += 1
                else:
                    merged.append(b[j])
                    j += 1
            while len(merged) < 10 and i < la:
                merged.append(a[i])
                i += 1
            while len(merged) < 10 and j < lb:
                merged.append(b[j])
                j += 1
            topk[ru] = merged
            # clear the merged-away root's list to save memory
            topk[rv] = []
        
        else:
            # Query k-th largest in component of v
            v = int(line[1])
            k_q = int(line[2])
            rv = find(v)
            if len(topk[rv]) >= k_q:
                answers.append(str(topk[rv][k_q - 1]))
            else:
                answers.append("-1")
    
    # Output all answers to type-2 queries
    sys.stdout.write("
".join(answers))

# Call main to execute
main()