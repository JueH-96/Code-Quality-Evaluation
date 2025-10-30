import math

def solve():
    n, q = map(int, input().split())
    h = list(map(int, input().split()))
    queries = []
    for _ in range(q):
        queries.append(tuple(map(int, input().split())))
    
    def is_visible(u_index, v_index):
        if u_index >= v_index:
            return False
        max_height_between = 0
        for k in range(u_index + 1, v_index):
            max_height_between = max(max_height_between, h[k-1])
        return max_height_between <= h[v_index-1]
        
    results = []
    for l_index, r_index in queries:
        count = 0
        for j_index in range(r_index + 1, n + 1):
            visible_from_l = is_visible(l_index, j_index)
            visible_from_r = is_visible(r_index, j_index)
            if visible_from_l and visible_from_r:
                count += 1
        results.append(count)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()