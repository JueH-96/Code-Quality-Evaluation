def can_make_abc(s):
    if s == 'abc':
        return "YES"
    swaps = [(0,1), (0,2), (1,2)]
    for i, j in swaps:
        # Swap characters at positions i and j
        s_list = list(s)
        s_list[i], s_list[j] = s_list[j], s_list[i]
        s_new = ''.join(s_list)
        if s_new == 'abc':
            return "YES"
    return "NO"

def main():
    import sys
    input = sys.stdin.read().splitlines()
    t = int(input[0])
    for i in range(1, t+1):
        s = input[i].strip()
        result = can_make_abc(s)
        print(result)

if __name__ == "__main__":
    main()