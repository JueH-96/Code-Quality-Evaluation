def solve():
    a, m, l, r = map(int, input().split())

    def count_trees(limit):
        if limit < a:
            return 0
        
        diff = limit - a
        return diff // m + 1

    
    result = count_trees(r) - count_trees(l - 1)
    print(result)

solve()