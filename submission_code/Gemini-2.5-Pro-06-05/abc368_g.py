import sys

def main():
    """
    Main function to read input, process queries, and print output.
    """
    # Fast I/O
    input = sys.stdin.readline
    sys.setrecursionlimit(2 * 10**5)

    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    Q = int(input())

    # The number of multiplications by a factor > 1 is at most ~60.
    # We choose K slightly larger for safety.
    K = 62

    # --- Fenwick Tree for sums of A ---
    ft = [0] * (N + 1)

    def ft_update(i, delta):
        i += 1
        while i <= N:
            ft[i] += delta
            i += i & -i

    def ft_query(i):
        i += 1
        s = 0
        while i > 0:
            s += ft[i]
            i -= i & -i
        return s
    
    def ft_query_range(l, r):
        if l > r:
            return 0
        res = ft_query(r)
        if l > 0:
            res -= ft_query(l - 1)
        return res

    for i in range(N):
        ft_update(i, A[i])

    # --- Segment Tree for finding last K multipliers ---
    st = [[] for _ in range(4 * N)]

    def merge_st(list1, list2):
        if not list1: return list2
        if not list2: return list1
        # A simple merge and truncate is sufficient as lists are sorted
        res = []
        i, j = 0, 0
        while i < len(list1) and j < len(list2) and len(res) < K:
            if list1[i] < list2[j]:
                res.append(list1[i])
                i += 1
            elif list2[j] < list1[i]:
                res.append(list2[j])
                j += 1
            else:
                res.append(list1[i])
                i += 1
                j += 1
        while i < len(list1) and len(res) < K:
            res.append(list1[i])
            i += 1
        while j < len(list2) and len(res) < K:
            res.append(list2[j])
            j += 1
        return res[-K:]

    def build_st(v, tl, tr):
        if tl == tr:
            if B[tl] > 1:
                st[v] = [tl]
        else:
            tm = (tl + tr) // 2
            build_st(2 * v, tl, tm)
            build_st(2 * v + 1, tm + 1, tr)
            st[v] = sorted(list(set(st[2*v] + st[2*v+1])))[-K:]
            
    build_st(1, 0, N - 1)

    def update_st(v, tl, tr, pos):
        if tl == tr:
            if B[pos] > 1:
                st[v] = [pos]
            else:
                st[v] = []
        else:
            tm = (tl + tr) // 2
            if pos <= tm:
                update_st(2 * v, tl, tm, pos)
            else:
                update_st(2 * v + 1, tm + 1, tr, pos)
            st[v] = sorted(list(set(st[2*v] + st[2*v+1])))[-K:]
            
    def query_st(v, tl, tr, l, r):
        if l > r or l > tr or r < tl:
            return []
        if l <= tl and tr <= r:
            return st[v]
        tm = (tl + tr) // 2
        left_res = query_st(2 * v, tl, tm, l, r)
        right_res = query_st(2 * v + 1, tm + 1, tr, l, r)
        return sorted(list(set(left_res + right_res)))[-K:]

    # --- Process Queries ---
    for _ in range(Q):
        query = list(map(int, input().split()))
        q_type = query[0]

        if q_type == 1:
            i, x = query[1] - 1, query[2]
            delta = x - A[i]
            A[i] = x
            ft_update(i, delta)
        elif q_type == 2:
            i, x = query[1] - 1, query[2]
            B[i] = x
            update_st(1, 0, N - 1, i)
        else: # q_type == 3
            l, r = query[1] - 1, query[2] - 1
            
            multipliers = query_st(1, 0, N - 1, l, r)

            if not multipliers:
                print(ft_query_range(l, r))
                continue

            p_start = multipliers[0]
            v_base = ft_query_range(l, p_start - 1)

            k = len(multipliers)
            dp = [[-1] * (k + 1) for _ in range(k)]

            dp[0][0] = v_base + A[p_start]
            dp[0][1] = v_base * B[p_start]
            
            for i in range(1, k):
                p_prev = multipliers[i - 1]
                p_curr = multipliers[i]
                
                sum_A_between = ft_query_range(p_prev + 1, p_curr - 1)

                for j in range(i + 2):
                    val_from_add, val_from_mult = -1, -1
                    if j <= i and dp[i - 1][j] != -1:
                        prev_v_add = dp[i-1][j] + sum_A_between
                        val_from_add = prev_v_add + A[p_curr]

                    if j > 0 and j - 1 <= i and dp[i-1][j-1] != -1:
                        prev_v_mult = dp[i-1][j-1] + sum_A_between
                        val_from_mult = prev_v_mult * B[p_curr]
                    
                    dp[i][j] = max(val_from_add, val_from_mult)
            
            max_v = 0
            for j in range(k + 1):
                if dp[k-1][j] != -1:
                    max_v = max(max_v, dp[k-1][j])
            
            max_v += ft_query_range(multipliers[-1] + 1, r)
            print(max_v)

if __name__ == "__main__":
    main()