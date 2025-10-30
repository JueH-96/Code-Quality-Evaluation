def isSubsequence(sub, main):
    sub_index = 0
    main_index = 0
    while sub_index < len(sub) and main_index < len(main):
        if sub[sub_index] == main[main_index]:
            sub_index += 1
        main_index += 1
    return sub_index == len(sub)

def solve():
    n, t = input().split()
    n = int(n)
    s_list = [input() for _ in range(n)]
    
    prefix_subseq = [[False] * (len(t) + 1) for _ in range(n)]
    suffix_subseq = [[False] * (len(t) + 1) for _ in range(n)]
    
    for i in range(n):
        for k in range(len(t) + 1):
            prefix_t = t[:k]
            prefix_subseq[i][k] = isSubsequence(prefix_t, s_list[i])
            
    for j in range(n):
        for k in range(len(t) + 1):
            suffix_t = t[k:]
            suffix_subseq[j][k] = isSubsequence(suffix_t, s_list[j])
            
    count = 0
    for i in range(n):
        for j in range(n):
            found_split = False
            for k in range(len(t) + 1):
                if prefix_subseq[i][k] and suffix_subseq[j][k]:
                    found_split = True
                    break
            if found_split:
                count += 1
                
    print(count)

if __name__ == '__main__':
    solve()