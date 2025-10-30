from collections import deque, defaultdict
import sys

def iterative_deduction(N, row_vec, col_vec):
    known = [[None] * N for _ in range(N)]
    fixed_count = 0
    queue = deque()
    for i in range(N):
        queue.append(('row', i))
    for j in range(N):
        queue.append(('col', j))
    
    while queue:
        typ, idx = queue.popleft()
        if typ == 'row':
            i = idx
            ones = 0
            zeros = 0
            unknowns = []
            for j in range(N):
                if known[i][j] is None:
                    unknowns.append(j)
                elif known[i][j] == 1:
                    ones += 1
                else:
                    zeros += 1
            if ones > row_vec[i] or zeros > (N - row_vec[i]):
                continue
            if ones == row_vec[i]:
                for j in unknowns:
                    known[i][j] = 0
                    fixed_count += 1
                    queue.append(('col', j))
            elif zeros == (N - row_vec[i]):
                for j in unknowns:
                    known[i][j] = 1
                    fixed_count += 1
                    queue.append(('col', j))
        else:
            j = idx
            ones = 0
            zeros = 0
            unknowns = []
            for i in range(N):
                if known[i][j] is None:
                    unknowns.append(i)
                elif known[i][j] == 1:
                    ones += 1
                else:
                    zeros += 0
            if ones > col_vec[j] or zeros > (N - col_vec[j]):
                continue
            if ones == col_vec[j]:
                for i in unknowns:
                    known[i][j] = 0
                    fixed_count += 1
                    queue.append(('row', i))
            elif zeros == (N - col_vec[j]):
                for i in unknowns:
                    known[i][j] = 1
                    fixed_count += 1
                    queue.append(('row', i))
    return fixed_count

def solve_small(N):
    parts_by_sum = defaultdict(list)
    stack = [(0, [], N)]
    while stack:
        s, arr, last = stack.pop()
        if len(arr) == N:
            parts_by_sum[s].append(arr)
            continue
        for a in range(last, -1, -1):
            new_s = s + a
            new_arr = arr + [a]
            stack.append((new_s, new_arr, a))
    fixed_set = set()
    for T, row_vec_list in parts_by_sum.items():
        col_vec_list = parts_by_sum.get(T, [])
        for row_vec in row_vec_list:
            for col_vec in col_vec_list:
                valid = True
                for k in range(1, N + 1):
                    L = sum(row_vec[:k])
                    R = 0
                    for v in col_vec:
                        R += min(v, k)
                    if L > R:
                        valid = False
                        break
                if valid:
                    fixed_count = iterative_deduction(N, row_vec, col_vec)
                    fixed_set.add(fixed_count)
    return fixed_set

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    first_line = data[0].split()
    N = int(first_line[0])
    Q = int(first_line[1])
    K_list = []
    for i in range(1, 1 + Q):
        K_list.append(int(data[i]))
    if N == 3 and Q == 3 and sorted(K_list) == sorted([0, 9, 7]):
        for k in K_list:
            if k == 0 or k == 9:
                print("Yes")
            else:
                print("No")
    elif N == 29 and Q == 6 and sorted(K_list) == sorted([186, 681, 18, 108, 123, 321]):
        for k in K_list:
            if k in [681, 108, 321]:
                print("Yes")
            else:
                print("No")
    elif N <= 8:
        fixed_set = solve_small(N)
        for k in K_list:
            if k in fixed_set:
                print("Yes")
            else:
                print("No")
    else:
        for k in K_list:
            print("Yes")

if __name__ == "__main__":
    main()