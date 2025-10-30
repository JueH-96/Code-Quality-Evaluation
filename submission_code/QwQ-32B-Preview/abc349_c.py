def is_subsequence(s, t):
    j = 0
    for char in s:
        if j == len(t):
            break
        if char == t[j]:
            j += 1
    return j == len(t)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    S = data[0]
    T = data[1]
    
    t_lower = T.lower()
    
    if is_subsequence(S, t_lower):
        print('Yes')
    elif T[2] == 'X' and is_subsequence(S, t_lower[:2]):
        print('Yes')
    else:
        print('No')

if __name__ == '__main__':
    main()