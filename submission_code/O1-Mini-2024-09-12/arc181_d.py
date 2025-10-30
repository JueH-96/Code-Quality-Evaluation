# YOUR CODE HERE
import sys, bisect

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    from bisect import bisect_left, insort

    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    P = list(map(int, data[idx:idx+N])); idx +=N
    M = int(data[idx]); idx +=1
    A = list(map(int, data[idx:idx+M])); idx +=M

    # Compute initial inversion count using BIT
    class BIT:
        def __init__(self, size):
            self.N = size
            self.tree = [0]*(self.N +2)
        def update(self, index, value=1):
            while index <= self.N:
                self.tree[index] += value
                index += index & -index
        def query(self, index):
            res =0
            while index >0:
                res += self.tree[index]
                index -= index & -index
            return res

    bit = BIT(N)
    initial_inversion =0
    for i in range(N-1, -1, -1):
        initial_inversion += bit.query(P[i]-1)
        bit.update(P[i],1)

    # Initialize S_list and S_set
    S_list = []
    S_set = set()
    for i in range(N-1):
        if P[i] > P[i+1]:
            S_list.append(i)
            S_set.add(i)
    S_list.sort()

    swap_count =0
    answers = []
    for p in range(M):
        k = A[p]
        # Find the index where i >=k
        idx_swap = bisect_left(S_list, k)
        swaps_to_perform = S_list[:idx_swap]
        S_list = S_list[idx_swap:]
        for i in swaps_to_perform:
            S_set.remove(i)
        for i in swaps_to_perform:
            # Swap P[i] and P[i+1}
            P[i], P[i+1] = P[i+1], P[i]
            swap_count +=1
            # Check i-1
            if i >0:
                if P[i-1] > P[i] and (i-1) not in S_set:
                    insort(S_list, i-1)
                    S_set.add(i-1)
            # Check i+1
            if i+1 <N-1:
                if P[i+1] > P[i+2] and (i+1) not in S_set:
                    insort(S_list, i+1)
                    S_set.add(i+1)
        inversion_now = initial_inversion - swap_count
        answers.append(str(inversion_now))
    print('
'.join(answers))

if __name__ == "__main__":
    main()