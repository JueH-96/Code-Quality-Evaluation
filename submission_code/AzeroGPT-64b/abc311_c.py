def main():
    from array import array

    N = int(input())
    a = array('i', map(int, input().split()))
    a = array('i', p - 1 for p in a)
    
    has_complete_cycle = array("b", [False] * N)
    result = []
    
    def dfs(start):
        nonlocal has_complete_cycle, result
        if has_complete_cycle[start]:
            return start
        else:
            has_complete_cycle[start] = True
            x = dfs(a[start])
            if x != -1:
                result.append(x + 1)
            return x if x == start else -1 if x == -1 else a[start]

    for p in range(N):
        if not has_complete_cycle[p]:
            dfs(p)
    
    print(len(result))
    print(' '.join(map(str, result[::-1])))

if __name__ == '__main__':
    main()