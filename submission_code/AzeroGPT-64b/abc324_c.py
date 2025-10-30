from bisect import bisect_left

def main():
    n, t = input().split()
    n = int(n)
    s_list = [input() for _ in range(n)]
    
    ans = []
    
    for i, s in enumerate(s_list):
        len_s = len(s)
        len_t = len(t)
        if len_s == len_t:
            if sum(a != b for a, b in zip(s, t)) > 1:
                continue
        elif len_s + 1 == len_t:
            if s in (t[:i] + t[i + 1:] for i in range(len_t)):
                ans.append(i)
                continue
        elif len_s - 1 == len_t:
            if t in (s[:i] + s[i:] for i in range(len_s)):
                ans.append(i)
                continue
        elif abs(len_s - len_t) > 1:
            pass
        else:
            if abs(sum(a != b for a, b in zip(s, t[:len_s]))) <= 1:
                ans.append(i)
                continue
                
    print(len(ans))
    print(' '.join(map(str, ans)))
    
main()