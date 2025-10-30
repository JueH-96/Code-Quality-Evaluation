def diff_count(s1, s2):
    return sum(1 for i in range(len(s1)) if s1[i] != s2[i])

def can_arrange(strings, used, curr_seq, n, m):
    if len(curr_seq) == n:
        return True
        
    for i in range(n):
        if not used[i]:
            if len(curr_seq) == 0 or diff_count(curr_seq[-1], strings[i]) == 1:
                used[i] = True
                curr_seq.append(strings[i])
                if can_arrange(strings, used, curr_seq, n, m):
                    return True
                used[i] = False
                curr_seq.pop()
    return False

def solve():
    n, m = map(int, input().split())
    strings = []
    for _ in range(n):
        strings.append(input().strip())
        
    used = [False] * n
    curr_seq = []
    
    if can_arrange(strings, used, curr_seq, n, m):
        print("Yes")
    else:
        print("No")

solve()