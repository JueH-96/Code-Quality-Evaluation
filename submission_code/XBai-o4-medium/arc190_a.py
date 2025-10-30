import sys
import bisect
import heapq

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [ (-float('inf'), -1) for _ in range(2 * self.size) ]
        # Fill the leaves
        for i in range(self.n):
            self.tree[self.size + i] = (data[i], i)
        # Build the tree
        for i in range(self.size - 1, 0, -1):
            left_val, left_idx = self.tree[2*i]
            right_val, right_idx = self.tree[2*i + 1]
            if left_val >= right_val:
                self.tree[i] = (left_val, left_idx)
            else:
                self.tree[i] = (right_val, right_idx)
    
    def update_val(self, pos, new_val):
        pos += self.size
        self.tree[pos] = (new_val, pos - self.size)
        pos >>= 1
        while pos >= 1:
            left_val, left_idx = self.tree[2*pos]
            right_val, right_idx = self.tree[2*pos + 1]
            if left_val >= right_val:
                self.tree[pos] = (left_val, left_idx)
            else:
                self.tree[pos] = (right_val, right_idx)
            pos >>= 1
    
    def query_range(self, l, r):
        # Query the maximum in [l, r], 0-based
        res = (-float('inf'), -1)
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                current_val, current_idx = self.tree[l]
                if current_val > res[0]:
                    res = (current_val, current_idx)
                l += 1
            if r % 2 == 0:
                current_val, current_idx = self.tree[r]
                if current_val > res[0]:
                    res = (current_val, current_idx)
                r -= 1
            l >>= 1
            r >>= 1
        return res

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    ops = []
    for i in range(M):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        ops.append( (L, R) )
    # Prepare sorted_ops
    sorted_ops = sorted( [ (ops[i][0], ops[i][1], i) for i in range(M) ], key=lambda x: x[0] )
    sorted_L = [ op[0] for op in sorted_ops ]
    # Build segment tree
    data_for_st = [ op[0] for op in sorted_ops ]
    st = SegmentTree(data_for_st)
    # Prepare priority queue
    pq = []
    j = 0
    used = [False] * M
    answer = [0] * M
    current = 1
    while current <= N:
        # Process case A
        caseA_reach = -float('inf')
        # Add all operations with Li <= current
        while j < len(sorted_ops) and sorted_ops[j][0] <= current:
            heapq.heappush(pq, ( -sorted_ops[j][1], j ))  # push (-Ri, index_in_sorted_ops)
            j += 1
        # Clean up the priority queue
        while True:
            if not pq:
                break
            neg_Ri, idx = pq[0]
            Ri = -neg_Ri
            op_in_sorted = sorted_ops[idx]
            if used[op_in_sorted[2]] or op_in_sorted[1] < current:
                # Pop it
                heapq.heappop(pq)
            else:
                break
        if pq:
            caseA_reach = -pq[0][0]
        # Process case B
        caseB_reach = -float('inf')
        current_plus_1 = current + 1
        low = bisect.bisect_left(sorted_L, current_plus_1)
        if low < len(sorted_L):
            max_Li, st_index = st.query_range(low, len(sorted_L)-1)
            if max_Li != -float('inf'):
                caseB_reach = max_Li - 1
        # Decide which case to choose
        best_reach = -float('inf')
        chosen_case = None
        if caseA_reach != -float('inf'):
            best_reach = caseA_reach
            chosen_case = 'A'
        if caseB_reach != -float('inf'):
            if caseB_reach > best_reach:
                best_reach = caseB_reach
                chosen_case = 'B'
            elif caseB_reach == best_reach:
                # Choose arbitrarily, say case A
                pass
        if best_reach == -float('inf'):
            print(-1)
            return
        if chosen_case == 'A':
            neg_Ri, idx = pq[0]
            Ri = -neg_Ri
            op_in_sorted = sorted_ops[idx]
            original_idx = op_in_sorted[2]
            answer[original_idx] = 1
            used[original_idx] = True
            # Update segment tree
            st.update_val(idx, -float('inf'))
            current = Ri + 1
        else:
            # Case B
            op_in_sorted = sorted_ops[st_index]
            original_idx = op_in_sorted[2]
            answer[original_idx] = 2
            used[original_idx] = True
            # Update segment tree
            st.update_val(st_index, -float('inf'))
            current = op_in_sorted[0]
    # After processing
    K = sum(answer)
    print(K)
    print(' '.join(map(str, answer)))

if __name__ == '__main__':
    main()