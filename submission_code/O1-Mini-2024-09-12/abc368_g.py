# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    Q = int(sys.stdin.readline())
    queries = [sys.stdin.readline().strip() for _ in range(Q)]
    
    # Segment tree size
    size = 1
    while size < N:
        size <<=1
    tree = [[] for _ in range(2*size)]
    
    def merge_functions(left, right):
        if not left:
            return right.copy()
        if not right:
            return left.copy()
        new_lines = []
        for m1, c1 in left:
            for m2, c2 in right:
                new_m = m1 * m2
                new_c = c1 * m2 + c2
                new_lines.append((new_m, new_c))
        # Sort by m ascending, if same m, keep higher c
        new_lines.sort(key=lambda x: (x[0], -x[1]))
        # Remove dominated lines
        envelope = []
        for line in new_lines:
            m, c = line
            if not envelope:
                envelope.append(line)
            elif envelope[-1][0] == m:
                if envelope[-1][1] < c:
                    envelope[-1] = line
            else:
                while len(envelope) >=2:
                    l1 = envelope[-2]
                    l2 = envelope[-1]
                    l3 = line
                    m1, c1 = l1
                    m2, c2 = l2
                    m3, c3 = l3
                    # Check if l2 is obsolete
                    if (c2 - c1)*(m1 - m3) <= (c3 - c1)*(m1 - m2):
                        envelope.pop()
                    else:
                        break
                envelope.append(line)
        return envelope

    # Build the segment tree
    for i in range(N):
        pos = size + i
        if B[i] ==1:
            tree[pos] = [(1, A[i])]
        else:
            if 1 == B[i]:
                tree[pos] = [(1, A[i])]
            else:
                if A[i] >=0:
                    tree[pos] = sorted([(1, A[i]), (B[i],0)], key=lambda x: (x[0], -x[1]))
                else:
                    tree[pos] = [(B[i],0), (1, A[i])]
    for i in range(size-1, 0, -1):
        left = tree[2*i]
        right = tree[2*i+1]
        tree[i] = merge_functions(left, right)

    # Update function
    def update_func(idx):
        pos = size + idx
        if B[idx] ==1:
            tree[pos] = [(1, A[idx])]
        else:
            if 1 == B[idx]:
                tree[pos] = [(1, A[idx])]
            else:
                if A[idx] >=0:
                    tree[pos] = sorted([(1, A[idx]), (B[idx],0)], key=lambda x: (x[0], -x[1]))
                else:
                    tree[pos] = [(B[idx],0), (1, A[idx])]
        pos >>=1
        while pos >=1:
            left = tree[2*pos]
            right = tree[2*pos+1]
            new = merge_functions(left, right)
            if new == tree[pos]:
                break
            tree[pos] = new
            pos >>=1

    # Query function
    def query(l, r):
        l += size
        r += size
        res_left = []
        res_right = []
        while l <= r:
            if l %2 ==1:
                res_left = merge_functions(res_left, tree[l])
                l +=1
            if r %2 ==0:
                res_right = merge_functions(tree[r], res_right)
                r -=1
            l >>=1
            r >>=1
        final = merge_functions(res_left, res_right)
        if not final:
            return 0
        return max(c for m,c in final)

    output = []
    for q in queries:
        parts = q.split()
        if parts[0] == '1':
            # Type 1: update A_i
            i = int(parts[1])-1
            x = int(parts[2])
            A[i] = x
            update_func(i)
        elif parts[0] == '2':
            # Type 2: update B_i
            i = int(parts[1])-1
            x = int(parts[2])
            B[i] = x
            update_func(i)
        else:
            # Type 3: query
            l = int(parts[1])-1
            r = int(parts[2])-1
            res = query(l, r)
            output.append(str(res))
    print('
'.join(output))

threading.Thread(target=main).start()