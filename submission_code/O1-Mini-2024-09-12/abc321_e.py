# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    from sys import stdin
    import sys

    def count_subtree(A, K, N):
        if K <0:
            return 0
        start = A << K
        end = (A +1) << K
        end -=1
        if start > N:
            return 0
        return min(end, N) - start +1

    data = sys.stdin.read().split()
    T = int(data[0])
    idx =1
    results = []
    for _ in range(T):
        N = int(data[idx])
        X = int(data[idx+1])
        K = int(data[idx+2])
        idx +=3
        anc = []
        A = X
        while A >=1:
            anc.append(A)
            A = A //2
        res =0
        for d, A in enumerate(anc):
            if d > K:
                break
            if d ==0:
                res += count_subtree(A, K, N)
            else:
                if d > K:
                    continue
                parent = anc[d]
                child_on_path = anc[d-1]
                if child_on_path == parent *2:
                    opp_child = parent *2 +1
                else:
                    opp_child = parent *2
                res += count_subtree(opp_child, K -d, N)
        results.append(str(res))
    print('
'.join(results))

if __name__ == "__main__":
    main()